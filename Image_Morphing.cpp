#include <opencv2\opencv.hpp>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <direct.h>
#pragma warning(disable : 4996)

// Change this value (Just 1~6, not zero)
#define SOURCE_IDX 5
#define TARGET_IDX 6

using namespace cv;
using namespace std;

// Classes for point(0D), edge(1D), triangle(2D)
typedef class point  // CLASS point (y,x)
{
public:
  pair<int, int> P;  // (y, x)

  point(int y, int x) { P = make_pair(y, x); }
  point(pair<int, int> p) : P(p) {}

  point operator - (point& x) { return point(make_pair(P.first - x.P.first, P.second - x.P.second)); }
  bool operator == (point& x) { return P == x.P; }

  long long norm2() { return (long long)P.first * P.first + (long long)P.second * P.second; }
} POINT;
typedef class edge   // CLASS edge (point to point; point idx)
{
public:
  pair<int, int> E;  // (point₁idx, point₂idx)

  edge(int a, int b) { E = make_pair(a, b); }
  edge(pair<int, int> e) : E(e) {}

  bool operator == (edge& cmp) { return E == cmp.E || (E.first == cmp.E.second && E.second == cmp.E.first); }  // Same edge(direction x)?
} EDGE;
typedef class triangle  // CLASS triangle (3 points; point idx)
{
public:
  // (point₁idx, point₂idx, point₃idx)
  int a;
  int b;
  int c;

  triangle(const triangle& t) { *this = t; }
  triangle(int i, int j, int k) : a(i), b(j), c(k) {}
} TRIANGLE;

// For convenience input/output
typedef struct data
{
  char txt_name[40], png_name[40];
  int h, w;
}DATA;
const DATA img_data[6] =
{
  {"./labels/1_600x600.txt", "./images/1_600x600.png", 600, 600},
  {"./labels/2_600x600.txt", "./images/2_600x600.png", 600, 600},
  {"./labels/3_480x480.txt", "./images/3_480x480.png", 480, 480},
  {"./labels/4_480x480.txt", "./images/4_480x480.png", 480, 480},
  {"./labels/5_500x500.txt", "./images/5_500x500.png", 500, 500},
  {"./labels/6_500x500.txt", "./images/6_500x500.png", 500, 500}
};
char dir[40];

int init(Mat&, vector<point>&, int); // Initialize func

int DelaunayTriangulation(vector<POINT>&, int, vector<TRIANGLE>&);  // O(N²) incremental Delaunay triangulation
bool inCircle(POINT&, POINT&, POINT&, POINT&);    // Determine whether point exists in the circumcircle or not
int CCW(POINT A, POINT B, POINT C)                    // Determine counterclockwise
{
  return (B.P.second - A.P.second) * (C.P.first - A.P.first) - (B.P.first - A.P.first) * (C.P.second - A.P.second);
}

void AffineTransform(vector<POINT>&, vector<POINT>&, vector<TRIANGLE>&, int, Mat&, Mat&);   // Affine transform for reverse mapping
void inverse3x3(double[][3], double[][3]);        // Get 3x3 inverse matrix
void multi3x3(double[][3], double[][3], double[][3]); // Multiply two 3x3 matrices
pair<int, int> getXinTri(POINT&, POINT&, POINT&, int);  // Get  all x coordinates in the triangle when given y coordinates
inline bool inTri(int alpha, int beta, int scale)  // AD = αAB + βAC (D is exist in triangle, iff 0≤α,β≤1 and 0≤α+β≤1 => Consider scale value!)
{
  return scale > 0 ? (alpha >= 0 && alpha <= scale && beta >= 0 && beta <= scale && alpha + beta >= 0 && alpha + beta <= scale)
    : (alpha <= 0 && alpha >= scale && beta <= 0 && beta >= scale && alpha + beta <= 0 && alpha + beta >= scale);
}

void crossDissolve(Mat&, Mat&, int);  // Make cross-dissolve image

void visualize(Mat&, vector<POINT>&, vector<TRIANGLE>&, int, int, int, int);  // Visualize triangle mesh

int main()
{
  int np;
  vector<POINT> vp1, vp2;          // Feature point vector
  Mat source, target;

  // Initialize
  if (init(source, vp1, SOURCE_IDX - 1) != init(target, vp2, TARGET_IDX - 1)) // Check # of feature points
  {
    puts("# of feature points in the source and target image is different!");
    return -1;
  }
  np = (int)vp1.size();

  for (int i = 1; i < 4; i++)     // 25%, 50%, 75% in-between image
  {
    sprintf(dir, "./%dvs%d_%d%%", SOURCE_IDX, TARGET_IDX, i * 25);
    mkdir(dir);

    vector<POINT> avg;            // Weighted average feature point
    vector<TRIANGLE> vtri;        // Triangle vector

    for (int j = 0; j < np; j++)  // Get the average feature points for both images
    {
      int y = ((4 - i) * vp1[j].P.first + i * vp2[j].P.first + 2) >> 2, x = ((4 - i) * vp1[j].P.second + i * vp2[j].P.second + 2) >> 2;
      avg.push_back(POINT(y, x));
    }

    // Apply incremental Delaunay triangulation to avg points
    DelaunayTriangulation(avg, np, vtri);
    visualize(source, vp1, vtri, SOURCE_IDX, SOURCE_IDX, TARGET_IDX, i);
    visualize(target, vp2, vtri, TARGET_IDX, SOURCE_IDX, TARGET_IDX, i);

    // Apply Affine transform & warping
    Mat s_transform(img_data[SOURCE_IDX - 1].h, img_data[SOURCE_IDX - 1].w, CV_8UC3);
    Mat t_transform(img_data[TARGET_IDX - 1].h, img_data[TARGET_IDX - 1].w, CV_8UC3);
    AffineTransform(vp1, avg, vtri, SOURCE_IDX - 1, source, s_transform);
    AffineTransform(vp2, avg, vtri, TARGET_IDX - 1, target, t_transform);

    // Apply cross dissolve to transformed images
    crossDissolve(s_transform, t_transform, i);
  }
  return 0;
}

int init(Mat& m, vector<point>& P, int idx)
{
  int N = 0;
  char str[15] = { 0, };

  // Read feature point text & img file
  m = imread(img_data[idx].png_name);
  ifstream ifs(img_data[idx].txt_name);
  if (!ifs)
  {
    puts("FILE OPEN ERROR");
    exit(-1);
  }

  for (ifs.getline(str, sizeof(str)); ifs.getline(str, sizeof(str)); N++)  // Get coordinate of points from file
  {
    if (strlen(str) < 5)
      break;

    int y = 0, x = 0, i;
    // Get matching points of img
    for (i = 0; str[i] != ','; i++)          // y
      y = y * 10 + str[i] - '0';
    for (++i; str[i] != ','; i++)            // x
      x = x * 10 + str[i] - '0';
    P.push_back(POINT(make_pair(y, x)));    // (y, x)
  }
  ifs.close();

  return N;
}
int DelaunayTriangulation(vector<POINT>& P, int N, vector<TRIANGLE>& T)
{
  // Already know supertriangle
  T.push_back(TRIANGLE(0, 1, 2));
  T.push_back(TRIANGLE(0, 2, 3));

  for (int i = 4; i < N; i++) // Start with the point inside the image
  {
    vector<EDGE> E;
    int nt = (int)T.size();

    for (int j = 0; j < nt; j++)
    {
      TRIANGLE* t = &T[j];
      if (inCircle(P[t->a], P[t->b], P[t->c], P[i]))  // Find all the triangles that include the new points in the circumcircle
      {
        // New triangle edges
        E.push_back(EDGE(t->a, t->b));
        E.push_back(EDGE(t->b, t->c));
        E.push_back(EDGE(t->c, t->a));

        T.erase(T.begin() + j--); // Delete triangle
        nt--; // Count down
      }
    }

    // Check duplicated edge
    int ne = (int)E.size();
    memset(check, 0, np * np);
    for (int j = 0; j < ne; j++)
      check[E[j].E.first * np + E[j].E.second]++; // Count up

    for (int j = 0; j < ne; j++)
      if (check[E[j].E.first * np + E[j].E.second] + check[E[j].E.second * np + E[j].E.first] == 1) // Unique edge
        T.push_back(TRIANGLE(E[j].E.first, E[j].E.second, i));  // Add new triangles
  }

  /*
  // Remove a triangle containing the vertex of the supertriangle (not using supertriangle)
  int nt = (int)T.size();
  for (int i = 0; i < nt; i++)
    if (T[i].a >= N || T[i].b >= N || T[i].c >= N)  // Contain supertriangle
    {
      T.erase(T.begin() + i--); // Delete
      nt--; // Count down
    }
  */

  delete[] check;
  return (int)T.size(); // Return # of triangles
  }

  /*
  // Remove a triangle containing the vertex of the supertriangle (not using supertriangle)
  int nt = (int)T.size();
  for (int i = 0; i < nt; i++)
    if (T[i].a >= N || T[i].b >= N || T[i].c >= N)  // Contain supertriangle
    {
      T.erase(T.begin() + i--); // Delete
      nt--; // Count down
    }
  */
  return (int)T.size(); // Return # of triangles
}
bool inCircle(POINT& A, POINT& B, POINT& C, POINT& D)
{
  POINT AD = A - D, BD = B - D, CD = C - D;   // Get 3 vectors
  int ccw = CCW(A, B, C);                     // Counterclockwise
  long long det = (long long)AD.P.second * (BD.P.first * CD.norm2() - CD.P.first * BD.norm2()) -
    (long long)AD.P.first * (BD.P.second * CD.norm2() - CD.P.second * BD.norm2()) +
    AD.norm2() * (long long)(BD.P.second * CD.P.first - CD.P.second * BD.P.first);  // Determinant

  return ccw > 0 ? det > 0 : det < 0; // Returns the sign of det depending on whether A, B, and C points are located counterclockwise
}

void AffineTransform(vector<POINT>& source, vector<POINT>& dest, vector<TRIANGLE>& T, int idx, Mat& origin, Mat& output)
{
  int nt = (int)T.size();
  double(*coeff)[3][3] = new double[nt][3][3];

  // Affine transform
  for (int i = 0; i < nt; i++)
  {
    // Make coordinates in matrix form
    double p_source[3][3] =
    {
      {(double)source[T[i].a].P.second, (double)source[T[i].a].P.first, 1},
      {(double)source[T[i].b].P.second, (double)source[T[i].b].P.first, 1},
      {(double)source[T[i].c].P.second, (double)source[T[i].c].P.first, 1}
    };
    double p_dest[3][3] =
    {
      {(double)dest[T[i].a].P.second, (double)dest[T[i].a].P.first, 1},
      {(double)dest[T[i].b].P.second, (double)dest[T[i].b].P.first, 1},
      {(double)dest[T[i].c].P.second, (double)dest[T[i].c].P.first, 1}
    };
    double temp[3][3], coefficient[3][3];

    // Get reverse mapping coefficient
    inverse3x3(p_source, temp);
    multi3x3(temp, p_dest, coefficient);
    inverse3x3(coefficient, coeff[i]);
  }

  // 2D dynamic memory alloc
  bool** check = new bool* [img_data[idx].h] { NULL, };
  check[0] = new bool[img_data[idx].h * img_data[idx].w]();

  for (int i = 1; i < img_data[idx].h; i++)
    check[i] = check[i - 1] + img_data[idx].w;

  // Warping
  for (int t = 0; t < nt; t++)
  {
    int yA = dest[T[t].a].P.first, yB = dest[T[t].b].P.first, yC = dest[T[t].c].P.first;
    int max = yA > yB ? yA > yC ? yA : yC : yB > yC ? yB : yC, min = yA < yB ? yA < yC ? yA : yC : yB < yC ? yB : yC;

    for (int i = min; i <= max; i++)
    {
      pair<int, int> X = getXinTri(dest[T[t].a], dest[T[t].b], dest[T[t].c], i);  // X : <start, end>
      int start = X.first < 0 ? 0 : X.first, end = X.second >= img_data[idx].w ? img_data[idx].w - 1 : X.second;

      for (int j = start; j <= end; j++)
      {
        if (!check[i][j])  // Not matched?
        {
          double x = coeff[t][0][0] * j + coeff[t][1][0] * i + coeff[t][2][0], y = coeff[t][0][1] * j + coeff[t][1][1] * i + coeff[t][2][1];
          int qx1 = x, qx2 = x + 1, qy1 = y, qy2 = y + 1;
          double alpha_x = qx2 - x, beta_x = x - qx1, alpha_y = qy2 - y, beta_y = y - qy1;

          // Handling exceptions outside the image
          if (qx1 < 0 || qx2 >= img_data[idx].w)
          {
            qx1 = qx2 = x < 0 ? 0 : img_data[idx].w - 1;
            alpha_x = beta_x = 0.5;
          }
          if (qy1 < 0 || qy2 >= img_data[idx].h)
          {
            qy1 = qy2 = y < 0 ? 0 : img_data[idx].h - 1;
            alpha_y = beta_y = 0.5;
          }

          /*output.at<Vec3b>(i, j) = alpha_x * alpha_y * origin.at<Vec3b>(qy1, qx1) + alpha_x * beta_y * origin.at<Vec3b>(qy2, qx1) +
            beta_x * alpha_y * origin.at<Vec3b>(qy1, qx2) + beta_x * beta_y * origin.at<Vec3b>(qy2, qx2);*/
          output.ptr<Vec3b>(i)[j] = alpha_x * alpha_y * origin.ptr<Vec3b>(qy1)[qx1] + alpha_x * beta_y * origin.ptr<Vec3b>(qy2)[qx1] +
            beta_x * alpha_y * origin.ptr<Vec3b>(qy1)[qx2] + beta_x * beta_y * origin.ptr<Vec3b>(qy2)[qx2];  // Bilinear interpolation
          check[i][j] = true;  // Pixel value matching complete
        }
      }
    }
  }

  // Visualize an image by making it a file
  char s[40];
  sprintf(s, "%s/Warp_%d.png", dir, idx + 1);
  imwrite(s, output);
  printf("Warping image file name : %s\n", s);

  delete[] coeff;
  delete[] check[0];
  delete[] check;
}
void inverse3x3(double source[][3], double dest[][3])
{
  // Get 3x3 inverse matrix
  dest[0][0] = source[1][1] * source[2][2] - source[1][2] * source[2][1];
  dest[0][1] = source[0][2] * source[2][1] - source[0][1] * source[2][2];
  dest[0][2] = source[0][1] * source[1][2] - source[0][2] * source[1][1];

  dest[1][0] = source[1][2] * source[2][0] - source[1][0] * source[2][2];
  dest[1][1] = source[0][0] * source[2][2] - source[0][2] * source[2][0];
  dest[1][2] = source[0][2] * source[1][0] - source[0][0] * source[1][2];

  dest[2][0] = source[1][0] * source[2][1] - source[1][1] * source[2][0];
  dest[2][1] = source[0][1] * source[2][0] - source[0][0] * source[2][1];
  dest[2][2] = source[0][0] * source[1][1] - source[0][1] * source[1][0];

  double det = source[0][0] * (source[1][1] * source[2][2] - source[1][2] * source[2][1]) +
    source[0][1] * (source[1][2] * source[2][0] - source[1][0] * source[2][2]) +
    source[0][2] * (source[1][0] * source[2][1] - source[1][1] * source[2][0]);

  // Inverse matrix is exist iff det≠0
  if (abs(det) < 1e-6)
  {
    puts("Inverse matrix doesn't exist");
    exit(-1);
  }

  // Divide scaled value(det)
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      dest[i][j] /= det;
}
void multi3x3(double a[][3], double b[][3], double c[][3])
{
  // Multiply two 3x3 matrices 
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    {
      double sum = 0;
      for (int k = 0; k < 3; k++)
        sum += a[i][k] * b[k][j];
      c[i][j] = sum;
    }
}
pair<int, int> getXinTri(POINT& A, POINT& B, POINT& C, int y)
{
  int yA = A.P.first, yB = B.P.first, yC = C.P.first, xA = A.P.second, xB = B.P.second, xC = C.P.second;

  vector<pair<int, bool>> x;
  int det = (xB - xA) * (yC - yA) - (xC - xA) * (yB - yA), alpha, beta;

  // AD = αAB + βAC (D is exist in triangle, iff 0≤α,β≤1 and 0≤α+β≤1)
  if (yA != yB)     // Search the intersection of line AB and y = c
  {
    int t = (y - yA) * (xB - xA) + (yB - yA) * xA; // x-coordinate of an intersection ((yB-yA) scaled)
    alpha = (yC - yA) * (t - xA * (yB - yA)) + (yB - yA) * (xA - xC) * (y - yA);  // det scaled α
    beta = (yA - yB) * (t - xA * (yB - yA)) + (yB - yA) * (xB - xA) * (y - yA);  // det scaled β

    if (inTri(alpha, beta, (yB - yA) * det))  // Intersection exists in the triangle
      x.push_back(make_pair(t / (yB - yA), t % (yB - yA) != 0));
  }
  if (yB != yC)     // Search the intersection of line BC and y = c
  {
    int t = (y - yB) * (xC - xB) + (yC - yB) * xB; // x-coordinate of an intersection ((yC-yB) scaled)
    alpha = (yC - yA) * (t - xA * (yC - yB)) + (yC - yB) * (xA - xC) * (y - yA);  // det scaled α
    beta = (yA - yB) * (t - xA * (yC - yB)) + (yC - yB) * (xB - xA) * (y - yA);  // det scaled β

    if (inTri(alpha, beta, (yC - yB) * det)) // Intersection exists in the triangle
      x.push_back(make_pair(t / (yC - yB), t % (yC - yB) != 0));
  }
  if (yC != yA)     // Search the intersection of line CA and y = c
  {
    int t = (y - yC) * (xA - xC) + (yA - yC) * xC; // x-coordinate of an intersection ((yA-yC) scaled)
    alpha = (yC - yA) * (t - xA * (yA - yC)) + (yA - yC) * (xA - xC) * (y - yA);  // det scaled α
    beta = (yA - yB) * (t - xA * (yA - yC)) + (yA - yC) * (xB - xA) * (y - yA);  // det scaled β

    if (inTri(alpha, beta, (yA - yC) * det)) // Intersection exists in the triangle
      x.push_back(make_pair(t / (yA - yC), t % (yA - yC) != 0));
  }

  if (x.size() < 2)          // Unknown error
    return make_pair(2, 1);  // Do not transform

  sort(x.begin(), x.end());
  int start = x[0].first + (x[0].second == true), end = x[x.size() - 1].first;
  return make_pair(start, end);  // x in [start, end]
}

void crossDissolve(Mat& source, Mat& target, int alpha)
{
  Mat cross = (((4 - alpha) * source) + alpha * target) / 4;  // Linear interpolation

  // Visualize morphed image by making it a file
  char s[40];
  sprintf(s, "%s/Morph.png", dir);
  imwrite(s, cross);
  printf("Morphed image file name : %s\n\n", s);
}

void visualize(Mat& img, vector<POINT>& P, vector<TRIANGLE>& T, int idx, int s_idx, int t_idx, int alpha)
{
  char s[40];
  // Read source/target img & duplicate
  Mat output;
  img.copyTo(output);

  for (int i = 0; i < T.size(); i++)      // Draw triangles to copied img
  {
    line(output, Point(P[T[i].a].P.second, P[T[i].a].P.first),
      Point(P[T[i].b].P.second, P[T[i].b].P.first), Scalar::all(255), 2);
    line(output, Point(P[T[i].b].P.second, P[T[i].b].P.first),
      Point(P[T[i].c].P.second, P[T[i].c].P.first), Scalar::all(255), 2);
    line(output, Point(P[T[i].c].P.second, P[T[i].c].P.first),
      Point(P[T[i].a].P.second, P[T[i].a].P.first), Scalar::all(255), 2);

    //sprintf(s, "%d", i + 1);
    //putText(output, s, Point((P[T[i].a].P.second + P[T[i].b].P.second + P[T[i].c].P.second) / 3, (P[T[i].c].P.first + P[T[i].b].P.first + P[T[i].a].P.first) / 3), FONT_HERSHEY_PLAIN, 2, Scalar::all(0));
  }

  // Visualize an image by making it a file
  sprintf(s, "%s/Triangulation_%d.png", dir, idx);
  imwrite(s, output);
  printf("Triangulation image file name : %s\n", s);
}