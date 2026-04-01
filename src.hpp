// src.hpp - Handwritten digit recognition (heuristic, non-ML)
// The judge() function takes a 28x28 grayscale image (values in [0,1])
// and returns an integer 0..9 predicting the digit.
// Heuristic approach using simple geometric and topological features.

#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <utility>

typedef std::vector< std::vector<double> > IMAGE_T;

namespace nr_heur {

static inline int H() { return 28; }
static inline int W() { return 28; }

static inline bool inb(int r, int c) { return r>=0 && r<H() && c>=0 && c<W(); }

// C++03-compatible BFS helper at namespace scope
static std::pair<int,bool> bfs_func(const IMAGE_T &bimg, std::vector< std::vector<int> > &vism,
                                    int sr, int sc, int target, int RR, int CC){
    static const int dr[4]={1,-1,0,0};
    static const int dc[4]={0,0,1,-1};
    std::queue< std::pair<int,int> > q;
    q.push(std::make_pair(sr,sc));
    vism[sr][sc]=1; int cnt=0; bool touches_border=false;
    while(!q.empty()){
        std::pair<int,int> pr = q.front(); q.pop(); int r=pr.first, c=pr.second; cnt++;
        if (r==0||c==0||r==RR-1||c==CC-1) touches_border=true;
        for (int k=0;k<4;++k){ int nr=r+dr[k], nc=c+dc[k]; if (nr>=0&&nr<RR&&nc>=0&&nc<CC && !vism[nr][nc]){
            if (target==0 && bimg[nr][nc]<0.5) { vism[nr][nc]=1; q.push(std::make_pair(nr,nc)); }
            if (target==1 && bimg[nr][nc]>0.5) { vism[nr][nc]=1; q.push(std::make_pair(nr,nc)); }
        }}
    }
    return std::make_pair(cnt,touches_border);
}

struct Feat {
    int rows, cols;
    int pixels;                   // foreground count
    int bbox_r0, bbox_c0, bbox_r1, bbox_c1;
    double aspect;                // bbox aspect ratio
    int vert_strokes;             // vertical stroke count by column clustering
    int horiz_strokes;            // horizontal strokes by row clustering
    bool has_hole;                // contains enclosed hole
    int euler;                    // Euler characteristic (#components - #holes)
    int top_heavy;                // mass more top than bottom
    int left_heavy;               // mass more left than right
    int right_heavy;              // mass more right than left
    double v_sym;                 // vertical symmetry score (higher better)
    double h_sym;                 // horizontal symmetry score
    Feat(): rows(28), cols(28), pixels(0),
            bbox_r0(28), bbox_c0(28), bbox_r1(-1), bbox_c1(-1),
            aspect(1.0), vert_strokes(0), horiz_strokes(0),
            has_hole(false), euler(0), top_heavy(0), left_heavy(0), right_heavy(0),
            v_sym(0.0), h_sym(0.0) {}
};

static IMAGE_T binarize(const IMAGE_T &img, double thr=0.5) {
    IMAGE_T b(img.size(), std::vector<double>(img[0].size()));
    for (size_t i=0;i<img.size();++i) for (size_t j=0;j<img[i].size();++j)
        b[i][j] = (img[i][j] < thr) ? 1.0 : 0.0; // digit is darker (white per spec, but adopt robustness)
    return b;
}

static Feat compute_feat(const IMAGE_T &img_in) {
    const int R = (int)img_in.size();
    const int C = (int)img_in[0].size();
    IMAGE_T img = img_in;
    // If values are [0,1] where 1 is white (background) and 0 is black (digit), invert threshold accordingly.
    // Robust binarization: decide polarity by median.
    std::vector<double> vals; vals.reserve(R*C);
    for (int i=0;i<R;++i) for (int j=0;j<C;++j) vals.push_back(img[i][j]);
    std::nth_element(vals.begin(), vals.begin()+vals.size()/2, vals.end());
    double med = vals[vals.size()/2];
    bool digit_dark = med > 0.5; // if median high (bright background), digit should be dark
    IMAGE_T b(R, std::vector<double>(C,0));
    for (int i=0;i<R;++i) for (int j=0;j<C;++j) b[i][j] = (digit_dark ? (img[i][j] < 0.5) : (img[i][j] > 0.5)) ? 1.0 : 0.0;

    Feat f; f.rows=R; f.cols=C;
    // Foreground count and bbox
    for (int i=0;i<R;++i) for (int j=0;j<C;++j) if (b[i][j]>0.5) {
        f.pixels++;
        f.bbox_r0 = std::min(f.bbox_r0, i);
        f.bbox_c0 = std::min(f.bbox_c0, j);
        f.bbox_r1 = std::max(f.bbox_r1, i);
        f.bbox_c1 = std::max(f.bbox_c1, j);
    }
    if (f.bbox_r1>=f.bbox_r0 && f.bbox_c1>=f.bbox_c0) {
        int bh = f.bbox_r1 - f.bbox_r0 + 1;
        int bw = f.bbox_c1 - f.bbox_c0 + 1;
        f.aspect = (double)bw / std::max(1, bh);
    }

    // Symmetry scores
    {
        int cnt=0, same=0;
        for (int i=0;i<R;++i) for (int j=0;j<C/2;++j){ cnt++; if (b[i][j]==b[i][C-1-j]) same++; }
        f.v_sym = cnt? (double)same/cnt : 0.0;
    }
    {
        int cnt=0, same=0;
        for (int i=0;i<R/2;++i) for (int j=0;j<C;++j){ cnt++; if (b[i][j]==b[R-1-i][j]) same++; }
        f.h_sym = cnt? (double)same/cnt : 0.0;
    }

    // Mass distribution
    long long top=0,bottom=0,left=0,right=0;
    for (int i=0;i<R;++i) for (int j=0;j<C;++j) if (b[i][j]>0.5){
        if (i < R/2) top++; else bottom++;
        if (j < C/2) left++; else right++;
    }
    f.top_heavy = (top>bottom) ? 1 : ((top<bottom)?-1:0);
    f.left_heavy = (left>right) ? 1 : ((left<right)?-1:0);
    f.right_heavy = (right>left) ? 1 : ((right<left)?-1:0);

    // Stroke counts by simple run-length per column/row inside bbox
    // Count strokes via simple run-length in bbox
    int v_strokes=0; {
        bool in=false;
        for (int j=f.bbox_c0;j<=f.bbox_c1;++j){
            int colsum=0;
            for (int i=f.bbox_r0;i<=f.bbox_r1;++i) if (b[i][j]>0.5) colsum++;
            bool now = colsum > (int)((f.bbox_r1-f.bbox_r0+1)*0.2);
            if (now && !in){ v_strokes++; in=true; }
            if (!now) in=false;
        }
    }
    int h_strokes=0; {
        bool in=false;
        for (int i=f.bbox_r0;i<=f.bbox_r1;++i){
            int rowsum=0;
            for (int j=f.bbox_c0;j<=f.bbox_c1;++j) if (b[i][j]>0.5) rowsum++;
            bool now = rowsum > (int)((f.bbox_c1-f.bbox_c0+1)*0.2);
            if (now && !in){ h_strokes++; in=true; }
            if (!now) in=false;
        }
    }
    f.vert_strokes = v_strokes;
    f.horiz_strokes = h_strokes;

    // Connectivity and holes via flood fill on background within bbox padding
    std::vector< std::vector<int> > vis(R, std::vector<int>(C,0));
    std::pair<int,bool> bfs_res;
    // Count foreground components
    int comp=0; for (int i=0;i<R;++i) for (int j=0;j<C;++j) if (!vis[i][j] && b[i][j]>0.5){ bfs_res=bfs_func(b,vis,i,j,1,R,C); (void)bfs_res; comp++; }
    // Reset vis and find background components that are enclosed (holes)
    for (int i=0;i<R;++i) for (int j=0;j<C;++j) vis[i][j]=0;
    int holes=0; for (int i=0;i<R;++i) for (int j=0;j<C;++j) if (!vis[i][j] && b[i][j]<0.5){ std::pair<int,bool> pr=bfs_func(b,vis,i,j,0,R,C); if (!pr.second) holes++; }
    f.has_hole = (holes>0);
    f.euler = comp - holes;

    return f;
}

static int classify(const Feat &f){
    // Rule-based classification using common MNIST heuristics
    // 8 has two holes usually; 0,6,9 have one hole; 4 has open loop often; 1 is tall thin; 7 has top stroke and right-heavy; 2 has top-right curve and bottom-left tail; 3 right-heavy with one hole-like notch.

    // Large or tiny digits fall back to simple rules
    if (f.pixels < 20) return 1; // almost empty -> 1 as slender

    // Hole-based first
    if (f.has_hole) {
        if (f.euler <= -1) { // two holes -> likely 8
            return 8;
        }
        // one hole: 0,6,9
        // If left-heavy and hole upper region -> 6; if right-heavy and top-heavy -> 9; else 0
        if (f.left_heavy>0 && f.top_heavy>=0) return 6;
        if (f.right_heavy>0 && f.top_heavy>0) return 9;
        // Symmetry favors 0
        if (f.v_sym > 0.85 && f.h_sym > 0.75) return 0;
        return 0;
    }

    // No hole digits: 1,2,3,4,5,7
    // Very tall and thin -> 1
    if (f.aspect < 0.5 && f.vert_strokes==1) return 1;

    // Many horizontal strokes and top-heavy with right leaning -> 7
    if (f.horiz_strokes<=2 && f.vert_strokes>=2 && f.top_heavy>0 && f.right_heavy>0) return 7;

    // 4 tends to have a vertical and horizontal cross and is not very symmetric
    if (f.vert_strokes>=2 && f.horiz_strokes>=2 && f.v_sym < 0.7) return 4;

    // Distinguish 2,3,5 using symmetry and weight
    // 3 is right-heavy and somewhat symmetric horizontally
    if (f.right_heavy>0 && f.horiz_strokes>=2) return 3;
    // 5 is left-heavy with bottom mass
    if (f.left_heavy>0 && f.top_heavy<0) return 5;
    // 2 is top-heavy with left-to-right sweep
    if (f.top_heavy>0 && f.right_heavy>0) return 2;

    // Fallbacks based on aspect
    if (f.aspect >= 1.2) return 2; // wider shape could be 2 or 3
    if (f.aspect <= 0.6) return 1;

    // Default guess
    return 7;
}

} // namespace

int judge(IMAGE_T &img) {
    using namespace nr_heur;
    if (img.empty() || img[0].empty()) return 0;
    // If image size differs, rescale to 28x28 via nearest neighbor for robustness
    int R = (int)img.size();
    int C = (int)img[0].size();
    IMAGE_T norm(28, std::vector<double>(28,1.0));
    for (int i=0;i<28;++i){
        for (int j=0;j<28;++j){
            int si = (int)((double)i*(R-1)/27.0 + 0.5); if (si<0) si=0; if (si>R-1) si=R-1;
            int sj = (int)((double)j*(C-1)/27.0 + 0.5); if (sj<0) sj=0; if (sj>C-1) sj=C-1;
            norm[i][j] = img[si][sj];
        }
    }
    nr_heur::Feat f = nr_heur::compute_feat(norm);
    int ans = nr_heur::classify(f);
    if (ans<0 || ans>9) ans = 0;
    return ans;
}
