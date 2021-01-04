/*
Assignment-1 OpenMP
Roll Number 1: 17QE30008, Mridul Agarwal
Roll Number 2: 17QE30007, Aniruddha Chattopadhyay

Note 1: Angle is to be given in degrees

Note 2: Time taken by our program to run without OpenMP inclusion and command execution is much lesser.
However, we have submitted the file with OpenMP parallelism.

To run the code:
Compile: gcc -fopenmp -lm 17qe30008_17qe30007.c
Run: ./a.out number_of_threads axisFileName objectFileName angleOfRotation

Output file - 'output.txt' contains the coordinates of all the points of the object in the same sequence.
Time taken by the program is being printed.
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct Point {
    float x, y, z;
} point;

void multiply(float first[][4], float second[], float mult[], int r1, int c1, int r2, int c2, int nthreads) {
	float sum;
	// Multiplying matrix with vector and storing in mult
	int i, k;
    #pragma omp parallel shared(mult) private(i, k, sum) 
    {
        #pragma omp for schedule(static)
        for (i = 0; i < r1; ++i)
        {
            sum = 0;
            for (k = 0; k < c1; ++k)
            {
                sum += first[i][k] * second[k];
            }
            mult[i] = sum;
        }
    }
    return;
}

int main(int argc, char *argv[])
{
    double wtime;
    wtime = omp_get_wtime();
    char p1[40], q1[40];

    if (argc != 5)
    {
        printf("Incorrect number of arguments passed.\n");
        exit(0);
    }

    int n_threads = atoi(argv[1]);
    float angle_deg = atof(argv[4]);
    float angle = angle_deg * M_PI / 180;
    omp_set_num_threads(n_threads);

    FILE *axis;
    axis = fopen(argv[2], "r");
    FILE *object;
    object = fopen(argv[3], "r");

    fscanf(axis, "%s %s", p1, q1);

    char *x, *y, *z;
    x = (char *) malloc(sizeof(char) * 10);
    y = (char *) malloc(sizeof(char) * 10);
    z = (char *) malloc(sizeof(char) * 10);
    
    x = strtok(p1, ",");
    y = strtok(NULL, ",");
    z = strtok(NULL, ")");

    point p, q;
    x = x + 2;
    p.x = atof(x);
    p.y = atof(y);
    p.z = atof(z);

    x = strtok(q1, ",");
    y = strtok(NULL, ",");
    z = strtok(NULL, ")");

    x = x + 2;
    q.x = atof(x);
    q.y = atof(y);
    q.z = atof(z);

    point ax;
   	ax.x = p.x - q.x;
	ax.y = p.y - q.y;
	ax.z = p.z - q.z;

    point pt;
    FILE *output1;
    output1 = fopen("output.txt", "w");

    char c1, prevc;
    int lines;

    for (c1 = getc(object), lines = 0; c1 != EOF; prevc = c1, c1 = getc(object))
    {
        if (c1 == '\n')
        {
            ++lines;
        }
    }
    if (prevc != '\n')
        ++lines;

    // printf("%d\n", lines);
    
    point pts[lines];
    fseek(object, 0, SEEK_SET);
    int i;

    for (i = 0; i < lines; ++i)
    {
        fscanf(object, "%f %f %f", &pts[i].x, &pts[i].y, &pts[i].z);
    }
    
    float x1, y1, z1;
    x1 = p.x; y1 = p.y; z1 = p.z;

    float T[4][4] = {1,0,0,-x1,0,1,0,-y1,0,0,1,-z1,0,0,0,1};

    float T_inverse[4][4] = { 1,0,0,x1,0,1,0,y1,0,0,1,z1,0,0,0,1 };

    float P = sqrt(ax.x * ax.x + ax.y * ax.y + ax.z * ax.z);
    struct Point u;
    u.x = ax.x/P;
    u.y = ax.y/P;
    u.z = ax.z/P;
    float a, b, c;

    a = u.x;
    b = u.y;
    c = u.z;

    float d = sqrt(c * c + b*b);
    float Rx[4][4] = {1,0,0,0,0,c/d,-b/d,0,0,b/d,c/d,0,0,0,0,1};

    float Rx_inverse[4][4] = { 1,0,0,0,0,c / d,b / d,0,0,-b / d,c / d,0,0,0,0,1 };

    float Ry[4][4] = { d,0,-a,0,0,1,0,0,a,0,d,0,0,0,0,1 };

    float Ry_inverse[4][4] = { d,0,a,0,0,1,0,0,-a,0,d,0,0,0,0,1 };

    float sin_t = sin(angle);
    float cos_t = cos(angle);

    float Rz[4][4] = { cos_t,-sin_t,0,0,sin_t,cos_t,0,0,0,0,1,0,0,0,0,1 };

    float mult1[4], mult2[4], mult3[4], mult4[4], mult5[4], mult6[4], mult7[4];

    point results[lines];

    #pragma omp parallel shared(results) private(mult1, mult2, mult3, mult4, mult5, mult6, mult7, i)
    {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < lines; ++i)
        {

            float X[4] = { pts[i].x,pts[i].y,pts[i].z,1};
            multiply(T, X, mult1, 4, 4, 4, 1, n_threads);
            multiply(Rx, mult1, mult2, 4, 4, 4, 1, n_threads);
            multiply(Ry, mult2, mult3, 4, 4, 4, 1, n_threads);
            multiply(Rz, mult3, mult4, 4, 4, 4, 1, n_threads);
            multiply(Ry_inverse, mult4, mult5, 4, 4, 4, 1, n_threads);
            multiply(Rx_inverse, mult5, mult6, 4, 4, 4, 1, n_threads);
            multiply(T_inverse, mult6, mult7, 4, 4, 4, 1, n_threads);

            results[i].x = mult7[0];
            results[i].y = mult7[1];
            results[i].z = mult7[2];
        }
    }
    
    for (i = 0; i < lines; ++i)
    {
        fprintf(output1, "%.4f %.4f %.4f\n", results[i].x, results[i].y, results[i].z);
    }

    wtime = omp_get_wtime() - wtime;
    printf("Time taken: %f seconds\n", wtime);
    return 0;
}