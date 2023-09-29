# Computer-Vision_Image-Morphing-2023
특징점을 이용한 이미지 모핑 알고리즘 구현

  + 이미지 read, write에 openCV 라이브러리 사용
    - 이외 기능 구현에는 일체 외부 라이브러리 사용 x
  + Mesh warping 방식으로 이미지 모핑 구현
    - Triangular mesh 생성 - Affine transform - cross dissolve 순으로 진행
      * Triangular mesh는 $O(N^2)$의 incremental delaunay triangulation 사용
