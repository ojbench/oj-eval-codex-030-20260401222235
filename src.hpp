// src.hpp - Handwritten digit recognition (heuristic, non-ML)
// Implements judge(IMAGE_T&) using only core C++03 features, no includes/typedefs.

namespace nr_heur_c03 {

static inline int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }

struct Feat {
    int pixels;
    int bbox_r0, bbox_c0, bbox_r1, bbox_c1;
    double aspect;
    int vert_strokes;
    int horiz_strokes;
    int has_hole;
    int euler;
    int top_heavy;
    int left_heavy;
    int right_heavy;
    double v_sym;
    double h_sym;
    // Extra features to improve 4/7 detection
    int has_cross;             // cross-junction exists
    int top_row_max;           // max row sum in top third of bbox
    double bottom_ratio;       // fraction of pixels in bottom half
    double right_ratio;        // fraction of pixels in right third
    Feat(): pixels(0), bbox_r0(28), bbox_c0(28), bbox_r1(-1), bbox_c1(-1),
            aspect(1.0), vert_strokes(0), horiz_strokes(0), has_hole(0), euler(0),
            top_heavy(0), left_heavy(0), right_heavy(0), v_sym(0.0), h_sym(0.0),
            has_cross(0), top_row_max(0), bottom_ratio(0.0), right_ratio(0.0) {}
};

static Feat compute_feat(IMAGE_T &img){
    const int R = (int)img.size();
    const int C = (R>0)? (int)img[0].size() : 0;
    int binv[28][28];
    // Decide polarity by mean
    double sum=0.0; int cnt=0;
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){ sum += img[i][j]; cnt++; }
    double mean = cnt? (sum/cnt) : 0.5;
    int digit_dark = (mean > 0.5) ? 1 : 0; // if background bright, digit dark
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){
        double v = img[i][j];
        int fg = digit_dark ? (v < 0.5) : (v > 0.5);
        binv[i][j] = fg ? 1 : 0;
    }

    Feat f;
    // Foreground count and bbox
    for(int i=0;i<R;i++) for(int j=0;j<C;j++) if (binv[i][j]){
        f.pixels++;
        if (i < f.bbox_r0) f.bbox_r0=i;
        if (j < f.bbox_c0) f.bbox_c0=j;
        if (i > f.bbox_r1) f.bbox_r1=i;
        if (j > f.bbox_c1) f.bbox_c1=j;
    }
    if (f.bbox_r1>=f.bbox_r0 && f.bbox_c1>=f.bbox_c0){
        int bh = f.bbox_r1 - f.bbox_r0 + 1;
        int bw = f.bbox_c1 - f.bbox_c0 + 1;
        f.aspect = (double)bw / (bh>0? bh:1);
    }

    // Symmetry scores
    {
        int same=0, total=0;
        for(int i=0;i<R;i++) for(int j=0;j<C/2;j++){ total++; if (binv[i][j]==binv[i][C-1-j]) same++; }
        f.v_sym = total? (double)same/total : 0.0;
    }
    {
        int same=0, total=0;
        for(int i=0;i<R/2;i++) for(int j=0;j<C;j++){ total++; if (binv[i][j]==binv[R-1-i][j]) same++; }
        f.h_sym = total? (double)same/total : 0.0;
    }

    // Mass distribution
    int top=0,bottom=0,left=0,right=0;
    for(int i=0;i<R;i++) for(int j=0;j>C?0:0;j++); // no-op to avoid empty loop warning
    for(int i=0;i<R;i++) for(int j=0;j<C;j++) if (binv[i][j]){
        if (i < R/2) top++; else bottom++;
        if (j < C/2) left++; else right++;
    }
    f.top_heavy = (top>bottom)?1:((top<bottom)?-1:0);
    f.left_heavy = (left>right)?1:((left<right)?-1:0);
    f.right_heavy = (right>left)?1:((right<left)?-1:0);

    // Extra ratios
    int bottom_pixels=0, right_third_pixels=0;
    for(int i=0;i<R;i++) for(int j=0;j<C;j++) if (binv[i][j]){
        if (i >= R/2) bottom_pixels++;
        if (j >= (2*C)/3) right_third_pixels++;
    }
    f.bottom_ratio = f.pixels? (double)bottom_pixels / (double)f.pixels : 0.0;
    f.right_ratio = f.pixels? (double)right_third_pixels / (double)f.pixels : 0.0;

    // Stroke counts in bbox
    int v_strokes=0; if (f.bbox_c0<=f.bbox_c1){
        int in=0;
        for(int j=f.bbox_c0;j<=f.bbox_c1;j++){
            int colsum=0;
            for(int i=f.bbox_r0;i<=f.bbox_r1;i++) if (binv[i][j]) colsum++;
            int now = (colsum > ((f.bbox_r1-f.bbox_r0+1)*2)/10) ? 1 : 0; // >20%
            if (now && !in){ v_strokes++; in=1; }
            if (!now) in=0;
        }
    }
    int h_strokes=0; if (f.bbox_r0<=f.bbox_r1){
        int in=0;
        for(int i=f.bbox_r0;i<=f.bbox_r1;i++){
            int rowsum=0;
            for(int j=f.bbox_c0;j<=f.bbox_c1;j++) if (binv[i][j]) rowsum++;
            int now = (rowsum > ((f.bbox_c1-f.bbox_c0+1)*2)/10) ? 1 : 0;
            if (now && !in){ h_strokes++; in=1; }
            if (!now) in=0;
        }
    }
    f.vert_strokes = v_strokes;
    f.horiz_strokes = h_strokes;

    // Cross-junction detection and top-row stroke strength
    int cross=0; int topmax=0;
    if (f.bbox_r0<=f.bbox_r1 && f.bbox_c0<=f.bbox_c1){
        int top_third_end = f.bbox_r0 + (f.bbox_r1 - f.bbox_r0 + 1)/3;
        for(int i=f.bbox_r0;i<=f.bbox_r1;i++){
            int rowsum=0;
            for(int j=f.bbox_c0;j<=f.bbox_c1;j++){
                if (binv[i][j]){
                    rowsum++;
                    if (i>0 && i<R-1 && j>0 && j<C-1){
                        if (binv[i-1][j] && binv[i+1][j] && binv[i][j-1] && binv[i][j+1]) cross=1;
                    }
                }
            }
            if (i<=top_third_end && rowsum>topmax) topmax=rowsum;
        }
    }
    f.has_cross = cross;
    f.top_row_max = topmax;

    // Connectivity and holes via BFS with static queue arrays
    int vis[28][28];
    for(int i=0;i<R;i++) for(int j=0;j<C;j++) vis[i][j]=0;
    int comp=0;
    int qx[28*28], qy[28*28];
    for(int i=0;i<R;i++) for(int j=0;j<C;j++) if (!vis[i][j] && binv[i][j]){
        int head=0, tail=0; qx[tail]=i; qy[tail]=j; tail++; vis[i][j]=1;
        while(head<tail){ int r=qx[head], c=qy[head]; head++;
            int nr, nc;
            nr=r+1; nc=c; if (nr>=0&&nr<R&&nc>=0&&nc<C && !vis[nr][nc] && binv[nr][nc]){ vis[nr][nc]=1; qx[tail]=nr; qy[tail]=nc; tail++; }
            nr=r-1; nc=c; if (nr>=0&&nr<R&&nc>=0&&nc<C && !vis[nr][nc] && binv[nr][nc]){ vis[nr][nc]=1; qx[tail]=nr; qy[tail]=nc; tail++; }
            nr=r; nc=c+1; if (nr>=0&&nr<R&&nc>=0&&nc<C && !vis[nr][nc] && binv[nr][nc]){ vis[nr][nc]=1; qx[tail]=nr; qy[tail]=nc; tail++; }
            nr=r; nc=c-1; if (nr>=0&&nr<R&&nc>=0&&nc<C && !vis[nr][nc] && binv[nr][nc]){ vis[nr][nc]=1; qx[tail]=nr; qy[tail]=nc; tail++; }
        }
        comp++;
    }
    for(int i=0;i<R;i++) for(int j=0;j<C;j++) vis[i][j]=0;
    int holes=0;
    for(int i=0;i<R;i++) for(int j=0;j<C;j++) if (!vis[i][j] && !binv[i][j]){
        int head=0, tail=0; qx[tail]=i; qy[tail]=j; tail++; vis[i][j]=1; int touches=0;
        while(head<tail){ int r=qx[head], c=qy[head]; head++;
            if (r==0||c==0||r==R-1||c==C-1) touches=1;
            int nr, nc;
            nr=r+1; nc=c; if (nr>=0&&nr<R&&nc>=0&&nc<C && !vis[nr][nc] && !binv[nr][nc]){ vis[nr][nc]=1; qx[tail]=nr; qy[tail]=nc; tail++; }
            nr=r-1; nc=c; if (nr>=0&&nr<R&&nc>=0&&nc<C && !vis[nr][nc] && !binv[nr][nc]){ vis[nr][nc]=1; qx[tail]=nr; qy[tail]=nc; tail++; }
            nr=r; nc=c+1; if (nr>=0&&nr<R&&nc>=0&&nc<C && !vis[nr][nc] && !binv[nr][nc]){ vis[nr][nc]=1; qx[tail]=nr; qy[tail]=nc; tail++; }
            nr=r; nc=c-1; if (nr>=0&&nr<R&&nc>=0&&nc<C && !vis[nr][nc] && !binv[nr][nc]){ vis[nr][nc]=1; qx[tail]=nr; qy[tail]=nc; tail++; }
        }
        if (!touches) holes++;
    }
    f.has_hole = holes>0 ? 1 : 0;
    f.euler = comp - holes;

    return f;
}

static int classify(const Feat &f){
    if (f.pixels < 20) return 1;
    if (f.has_hole){
        if (f.euler <= -1) return 8; // two holes
        if (f.left_heavy>0 && f.top_heavy>=0) return 6;
        if (f.right_heavy>0 && f.top_heavy>0) return 9;
        if (f.v_sym > 0.85 && f.h_sym > 0.75) return 0;
        return 0;
    }
    // Strong 4 heuristic: cross present, right-side heavy, top heavier than bottom
    if (f.has_cross && f.right_ratio > 0.32 && f.bottom_ratio < 0.48) return 4;
    // Strong 7 heuristic: strong top bar and light bottom
    if (f.top_row_max > 10 && f.bottom_ratio < 0.30) return 7;
    if (f.aspect < 0.5 && f.vert_strokes==1) return 1;
    if (f.vert_strokes>=2 && f.horiz_strokes>=2 && f.v_sym < 0.7 && f.right_ratio > 0.3) return 4;
    if (f.right_heavy>0 && f.horiz_strokes>=2) return 3;
    if (f.left_heavy>0 && f.top_heavy<0) return 5;
    if (f.top_heavy>0 && f.right_heavy>0 && f.h_sym < 0.6) return 2;
    if (f.aspect >= 1.2) return 2;
    if (f.aspect <= 0.6) return 1;
    return 7;
}

} // namespace

int judge(IMAGE_T &img){
    using namespace nr_heur_c03;
    if (img.size()==0 || img[0].size()==0) return 0;
    // Assume 28x28 per problem; use as-is
    Feat f = compute_feat(img);
    int ans = classify(f);
    if (ans<0 || ans>9) ans = 0;
    return ans;
}
