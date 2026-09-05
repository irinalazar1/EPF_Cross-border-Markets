import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ComponentProps, Streamlit } from "streamlit-component-lib";

interface Args {
  values: number[];
  labels: string[];
  radius: number;
  height: number;
}

const PAD_LEFT = 48;
const PAD_RIGHT = 16;
const PAD_TOP = 16;
const PAD_BOTTOM = 32;

/**
 * Linear falloff weight for how much a point at `distance` slots away from
 * the dragged point should move. 1.0 at distance 0, straight line down to
 * 0.0 at distance == radius, untouched beyond that.
 */
function falloffWeight(distance: number, radius: number): number {
  if (radius <= 0) return distance === 0 ? 1 : 0;
  const w = 1 - distance / radius;
  return w > 0 ? w : 0;
}

const DraggableCurve: React.FC<ComponentProps> = (props) => {
  const args = props.args as Args;
  const { labels, radius, height } = args;

  // `values` is the live, possibly-mid-drag array driving the chart.
  // `baseValuesRef` is a frozen snapshot taken at the moment a drag starts,
  // so every mousemove computes its delta from a fixed origin rather than
  // compounding small movements into drift.
  const [values, setValues] = useState<number[]>(args.values);
  const baseValuesRef = useRef<number[]>(args.values);
  const dragIndexRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const width = Math.max(props.width || 700, 300);
  const n = values.length;

  // Re-sync from Python if the incoming values genuinely changed (e.g. the
  // user picked a different forecast date) -- but not on every re-render,
  // or an in-progress drag would keep getting reset out from under itself.
  const argsValuesKey = args.values.join(",");
  const lastSyncedKeyRef = useRef<string>(argsValuesKey);
  useEffect(() => {
    if (argsValuesKey !== lastSyncedKeyRef.current) {
      lastSyncedKeyRef.current = argsValuesKey;
      setValues(args.values);
      baseValuesRef.current = args.values;
    }
  }, [argsValuesKey, args.values]);

  useEffect(() => {
    Streamlit.setComponentReady();
  }, []);

  useEffect(() => {
    Streamlit.setFrameHeight(height + PAD_TOP + PAD_BOTTOM + 24);
  }, [height]);

  // Axis range is frozen for the duration of an active drag, recomputed
  // only while idle. Without this, dragging one point to an extreme
  // rescales the whole y-axis live (since it's fit to current min/max on
  // every render) -- every OTHER point's pixel position then shifts too,
  // even though its actual value barely changed, which looks exactly like
  // "the other values moved the opposite way" even though they didn't.
  const axisRangeRef = useRef<[number, number]>([0, 1]);
  const isDragging = dragIndexRef.current !== null;

  const [yMin, yMax] = useMemo(() => {
    if (isDragging) {
      return axisRangeRef.current;
    }
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const pad = (hi - lo) * 0.15 || 1;
    const range: [number, number] = [lo - pad, hi + pad];
    axisRangeRef.current = range;
    return range;
  }, [values, isDragging]);

  const plotWidth = width - PAD_LEFT - PAD_RIGHT;
  const plotHeight = height - PAD_TOP - PAD_BOTTOM;

  const xForIndex = useCallback(
    (i: number) => PAD_LEFT + (n <= 1 ? 0 : (i / (n - 1)) * plotWidth),
    [n, plotWidth]
  );
  const yForValue = useCallback(
    (v: number) => PAD_TOP + (1 - (v - yMin) / (yMax - yMin || 1)) * plotHeight,
    [yMin, yMax, plotHeight]
  );
  const valueForY = useCallback(
    (y: number) => yMin + (1 - (y - PAD_TOP) / plotHeight) * (yMax - yMin),
    [yMin, yMax, plotHeight]
  );
  const indexForX = useCallback(
    (x: number) => {
      const raw = ((x - PAD_LEFT) / plotWidth) * (n - 1);
      return Math.min(n - 1, Math.max(0, Math.round(raw)));
    },
    [n, plotWidth]
  );

  const applyDrag = useCallback(
    (dragIndex: number, mouseY: number) => {
      const base = baseValuesRef.current;
      const newValueAtDrag = valueForY(mouseY);
      const delta = newValueAtDrag - base[dragIndex];

      const next = base.slice();
      for (let j = Math.max(0, dragIndex - radius); j <= Math.min(n - 1, dragIndex + radius); j++) {
        const weight = falloffWeight(Math.abs(j - dragIndex), radius);
        next[j] = base[j] + delta * weight;
      }
      setValues(next);
    },
    [n, radius, valueForY]
  );

  const getSvgPoint = (clientX: number, clientY: number) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: clientX - rect.left, y: clientY - rect.top };
  };

  const handlePointerDown = (clientX: number, clientY: number) => {
    const { x, y } = getSvgPoint(clientX, clientY);
    const idx = indexForX(x);
    dragIndexRef.current = idx;
    baseValuesRef.current = values.slice(); // freeze the starting shape for this drag
    applyDrag(idx, y);
  };

  const handlePointerMove = (clientX: number, clientY: number) => {
    if (dragIndexRef.current === null) return;
    const { y } = getSvgPoint(clientX, clientY);
    applyDrag(dragIndexRef.current, y);
  };

  const handlePointerUp = () => {
    if (dragIndexRef.current === null) return;
    dragIndexRef.current = null;
    lastSyncedKeyRef.current = values.join(","); // this IS the new "incoming" state now
    Streamlit.setComponentValue(values);
    setValues(values.slice()); // force a re-render now so the y-axis re-fits immediately, without waiting on the Python round-trip
  };

  // Refs always hold this render's freshest handler closures (correctly
  // capturing the current `values`, `radius`, etc.), so the window
  // listeners below can attach exactly ONCE for the component's lifetime
  // instead of being torn down and rebuilt on every mousemove during a
  // drag -- which is what happened when this effect depended on `values`.
  const handlePointerMoveRef = useRef(handlePointerMove);
  const handlePointerUpRef = useRef(handlePointerUp);
  handlePointerMoveRef.current = handlePointerMove;
  handlePointerUpRef.current = handlePointerUp;

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => handlePointerMoveRef.current(e.clientX, e.clientY);
    const onMouseUp = () => handlePointerUpRef.current();
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 0) {
        handlePointerMoveRef.current(e.touches[0].clientX, e.touches[0].clientY);
        e.preventDefault();
      }
    };
    const onTouchEnd = () => handlePointerUpRef.current();

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    window.addEventListener("touchmove", onTouchMove, { passive: false });
    window.addEventListener("touchend", onTouchEnd);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onTouchEnd);
    };
  }, []); // mount once -- handlers are always invoked via the refs above, so they never go stale

  const linePath = useMemo(() => {
    return values.map((v, i) => `${i === 0 ? "M" : "L"}${xForIndex(i)},${yForValue(v)}`).join(" ");
  }, [values, xForIndex, yForValue]);

  const labelStep = Math.max(1, Math.round(n / 8)); // ~8 x-axis labels regardless of n

  return (
    <div>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{ touchAction: "none", cursor: "ns-resize", display: "block" }}
        onMouseDown={(e) => handlePointerDown(e.clientX, e.clientY)}
        onTouchStart={(e) => {
          if (e.touches.length > 0) handlePointerDown(e.touches[0].clientX, e.touches[0].clientY);
        }}
      >
        {/* Invisible full-area hit target. SVG only registers pointer events
           on "painted" pixels by default (visiblePainted) -- without this,
           only the 2.5px line stroke and 3px circles themselves are
           clickable, which in practice is nearly impossible to hit. This
           rect makes the entire plot area draggable, not just those slivers. */}
        <rect
          x={PAD_LEFT}
          y={PAD_TOP}
          width={plotWidth}
          height={plotHeight}
          fill="transparent"
          style={{ pointerEvents: "all" }}
        />

        {/* y-axis gridlines + labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const v = yMin + t * (yMax - yMin);
          const y = yForValue(v);
          return (
            <g key={t}>
              <line x1={PAD_LEFT} y1={y} x2={width - PAD_RIGHT} y2={y} stroke="var(--gray-70, #444)" strokeWidth={0.5} opacity={0.35} />
              <text x={PAD_LEFT - 8} y={y} fontSize={11} textAnchor="end" dominantBaseline="middle" fill="var(--text-color, #888)">
                {v.toFixed(0)}
              </text>
            </g>
          );
        })}

        {/* x-axis labels */}
        {values.map((_, i) =>
          i % labelStep === 0 ? (
            <text key={i} x={xForIndex(i)} y={height - PAD_BOTTOM + 16} fontSize={10} textAnchor="middle" fill="var(--text-color, #888)">
              {labels[i]}
            </text>
          ) : null
        )}

        {/* the curve itself */}
        <path d={linePath} fill="none" stroke="#6366f1" strokeWidth={2.5} />

        {/* draggable point handles */}
        {values.map((v, i) => (
          <circle
            key={i}
            cx={xForIndex(i)}
            cy={yForValue(v)}
            r={dragIndexRef.current === i ? 6 : 4}
            fill="#6366f1"
            stroke="white"
            strokeWidth={1}
            style={{ cursor: "ns-resize" }}
          />
        ))}
      </svg>
      <div style={{ fontSize: 12, color: "var(--text-color, #888)", marginTop: 4 }}>
        Drag any point to adjust it — nearby points within {radius} slots shift too.
      </div>
    </div>
  );
};

export default DraggableCurve;
