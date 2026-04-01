#include "src.hpp"
#include <bits/stdc++.h>
using namespace std;
int main(){
    IMAGE_T img(28, vector<double>(28, 0.0)); // black background
    // draw a simple white vertical line (like '1')
    for(int i=3;i<25;i++) img[i][14]=1.0;
    int ans = judge(img);
    cout << ans << "\n";
    return 0;
}
