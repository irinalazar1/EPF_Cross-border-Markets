import React from "react";
import ReactDOM from "react-dom";
import { withStreamlitConnection } from "streamlit-component-lib";
import DraggableCurve from "./DraggableCurve";

const ConnectedDraggableCurve = withStreamlitConnection(DraggableCurve);

ReactDOM.render(
  <React.StrictMode>
    <ConnectedDraggableCurve />
  </React.StrictMode>,
  document.getElementById("root")
);
