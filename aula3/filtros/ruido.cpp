//ruido.cpp - pos2016
#include <cekeikon.h>

Mat_<GRY> ruido(Mat_<GRY> a) {
  Mat_<GRY> b=a.clone();
  srand(7);
  for (unsigned l=0; l<a.rows; l++) {
    for (unsigned c=0; c<a.cols; c++) {
      if (rand()%20==0) {
        b(l,c)=rand()%256;        
      }
    }
  }
  return b;
}

int main() {
  Mat_<GRY> a; le(a,"lennag.jpg");
  Mat_<GRY> b=ruido(a);
  imp(b,"ruido.png");
}
