Each directory is the date that the tests were run on (YYYYMMDD).
Each file has columns [Xim, Xiv, L, t, error]. Each row is one solver step towards the solution. 
    error := SSE of (hTarget - h(xim, xiv, L, t)) and (cTarget - c(xim, xiv, L, t))