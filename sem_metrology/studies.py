from __future__ import annotations

import numpy as np


def common_study_metrics(records: list[dict], minimum_records: int = 2) -> list[str]:
    counts: dict[str,int]={}
    for record in records:
        for metric,data in record.get("metrics",{}).items():
            if np.isfinite(data.get("mean",np.nan)): counts[metric]=counts.get(metric,0)+1
    return sorted(metric for metric,count in counts.items() if count>=minimum_records)


def dose_response_summary(records: list[dict], metric: str) -> dict:
    """Return a transparent linear response summary for one common metric."""
    points=[]
    for record in records:
        data=record.get("metrics",{}).get(metric,{})
        x=record.get("factor_value",np.nan); y=data.get("mean",np.nan)
        if np.isfinite(x) and np.isfinite(y):
            points.append((float(x),float(y),float(data.get("std",np.nan)),int(data.get("n",0))))
    if not points: return {"n":0}
    points=sorted(points); x=np.asarray([p[0] for p in points]); y=np.asarray([p[1] for p in points])
    result={"n":len(points),"x":x,"y":y,"std":np.asarray([p[2] for p in points]),
            "measurement_n":np.asarray([p[3] for p in points])}
    if len(np.unique(x))>=2:
        slope,intercept=np.polyfit(x,y,1); fitted=slope*x+intercept
        total=float(np.sum((y-y.mean())**2)); residual=float(np.sum((y-fitted)**2))
        result.update({"slope":float(slope),"intercept":float(intercept),
                       "r_squared":float(1-residual/total) if total>0 else np.nan,
                       "change":float(y[-1]-y[0]),
                       "percent_change":float((y[-1]-y[0])/abs(y[0])*100) if y[0] else np.nan})
    return result
