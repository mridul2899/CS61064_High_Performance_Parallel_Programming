#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "png.h"

#ifndef TRUE
#  define TRUE 1
#  define FALSE 0
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) > (b)? (a) : (b))
#  define MIN(a,b)  ((a) < (b)? (a) : (b))
#endif

#ifdef DEBUG
#  define Trace(x)  {fprintf x ; fflush(stderr); fflush(stdout);}
#else
#  define Trace(x)  ;
#endif

typedef unsigned char   uch;
typedef unsigned short  ush;
typedef unsigned long   ulg;

#ifndef png_jmpbuf
#  define png_jmpbuf(png_ptr)   ((png_ptr)->jmpbuf)
#endif

static png_structp png_ptr = NULL;
static png_infop info_ptr = NULL;

png_uint_32  width, height;
int  bit_depth, color_type;
uch  *image_data = NULL;

int readpng_init(FILE *infile, long *pWidth, long *pHeight);
uch *readpng_get_image(double display_exponent, int *pChannels, ulg *pRowbytes);
void readpng_cleanup(int free_image_data);

int main(int argc, char *argv[])
{
    double LUT_exponent;
    double CRT_exponent = 2.2;
    double default_display_exponent;
  
    #if defined(NeXT)
        LUT_exponent = 1.0 / 2.2;
        /*
        if (some_next_function_that_returns_gamma(&next_gamma))
            LUT_exponent = 1.0 / next_gamma;
        */
    #elif defined(sgi)
        LUT_exponent = 1.0 / 1.7;
        /* there doesn't seem to be any documented function to
        * get the "gamma" value, so we do it the hard way */
        infile = fopen("/etc/config/system.glGammaVal", "r");
        if (infile) {
            double sgi_gamma;
    
            fgets(fooline, 80, infile);
            fclose(infile);
            sgi_gamma = atof(fooline);
            if (sgi_gamma > 0.0)
                LUT_exponent = 1.0 / sgi_gamma;
        }
    #elif defined(Macintosh)
        LUT_exponent = 1.8 / 2.61;
        /*
        if (some_mac_function_that_returns_gamma(&mac_gamma))
            LUT_exponent = mac_gamma / 2.61;
        */
    #else
        LUT_exponent = 1.0;   /* assume no LUT:  most PCs */
    #endif

    default_display_exponent = LUT_exponent * CRT_exponent;
    char *p;
    double display_exponent;
    if ((p = getenv("SCREEN_GAMMA")) != NULL)
        display_exponent = atof(p);
    else
        display_exponent = default_display_exponent;
    
    FILE *image;
    image = fopen("sample1.png", "rb");
    long wid;
    long hei;
    int ret_val;
    ret_val = readpng_init(image, &wid, &hei);

    static ulg image_rowbytes;
    static int image_channels;
    image_data = readpng_get_image(display_exponent, &image_channels, &image_rowbytes);

    readpng_cleanup(FALSE);
    fclose(image);

    int grid_x, grid_y;
    int heig = hei / 2, widt = wid / 2;
    int start, stop;

    int rank, size;
    uch **image_mat;
    uch **new_image_mat;
    uch **sobel_image_mat;  

    uch *image_data_0;    
    image_data_0 = (uch *) malloc((hei * wid) * sizeof(uch));

    uch *image_data_1;    
    image_data_1 = (uch *) malloc((hei * wid) * sizeof(uch) / 4);

    image_mat = (uch **) malloc((heig + 1) * sizeof(uch *));
    for (int i = 0; i < heig + 1; ++i)
    {
        image_mat[i] = (uch *) malloc((widt + 1) * sizeof(uch));
    }
    new_image_mat = (uch **) malloc((heig + 1) * sizeof(uch *));
    for (int i = 0; i < heig + 1; ++i)
    {
        new_image_mat[i] = (uch *) malloc((widt + 1) * sizeof(uch));
    }
    sobel_image_mat = (uch **) malloc((heig) * sizeof(uch *));
    for (int i = 0; i < heig; ++i)
    {
        sobel_image_mat[i] = (uch *) malloc((widt) * sizeof(uch));
    }

    long values[256];
    long long cumm[256];
    uch new_pixel_values[256];

    int sobel_x[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    int sobel_y[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sob_val_x, sob_val_y;
    int sob_val;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0)
    {
        grid_x = 0;
        grid_y = 0;
    }
    else if (rank == 1)
    {
        grid_x = 0;
        grid_y = 1;
    }
    else if (rank == 2)
    {
        grid_x = 1;
        grid_y = 0;
    }
    else if (rank == 3)
    {
        grid_x = 1;
        grid_y = 1;
    }

    if (rank == 0)
    {
        for (int i = 0; i < hei * wid; ++i)
        {
            image_data_0[i] = image_data[(i * wid + j) * image_channels];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatter(image_data_0, hei * wid / 4, MPI_UNSIGNED_CHAR, image_data_1, hei * wid / 4, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < hei / 2, ++i)
    {
        for (int j)
    }
    for (int i = 0; i < 256; ++i)
    {
        values[i] = 0;
        cumm[i] = 0;
    }
    for (int i = 0; i < heig + 1; ++i)
    {
        for (int j = 0; j < widt + 1; ++j)
        {
            values[image_mat[i][j]] += 1;
        }
    }
    cumm[0] = values[0];
    for (int i = 1; i < 256; ++i)
    {
        cumm[i] = cumm[i - 1] + values[i];
    }    
    for (int i = 0; i < 256; ++i)
    {
        new_pixel_values[i] = (uch) (cumm[i] * 255 / ((widt + 1) * (hei + 1)));
    }
        
    for (int i = 0; i < heig + 1; ++i)
    {
        for (int j = 0; j < widt + 1; ++j)
        {
            new_image_mat[i][j] = new_pixel_values[image_mat[i][j]];
        }
    }

    MPI_Gather(image_data_0, hei * wid / 4, MPI_UNSIGNED_CHAR, image_data_1, hei * wid / 4, MPI_UNSIGNED_CHAR, rank, MPI_COMM_WORLD);

    for (int i = 0 + grid_x; i < heig + grid_x; ++i)
    {
        for (int j = 0 + grid_y; j < widt + grid_y; ++j)
        {
            sob_val_x = sob_val_y = 0;
            int a_min, a_max, b_min, b_max;
            if (i == 0)
            {
                if (j == 0)
                {
                    a_min = 1;
                    a_max = 3;
                    b_min = 1;
                    b_max = 3;
                }
                else if (j == widt)
                {
                    a_min = 1;
                    a_max = 3;
                    b_min = 0;
                    b_max = 2;
                }
                else
                {
                    a_min = 1;
                    a_max = 3;
                    b_min = 0;
                    b_max = 3;
                }
            }
            else if (i == heig)
            {
                if (j == 0)
                {
                    a_min = 0;
                    a_max = 2;
                    b_min = 1;
                    b_max = 3;
                }
                else if (j == widt)
                {
                    a_min = 0;
                    a_max = 2;
                    b_min = 0;
                    b_max = 2;
                }
                else
                {
                    a_min = 0;
                    a_max = 2;
                    b_min = 0;
                    b_max = 3;
                }    
            }
            else if (j == 0)
            {
                a_min = 0;
                a_max = 3;
                b_min = 1;
                b_max = 3;
            }
            else if (j == widt)
            {
                a_min = 0;
                a_max = 3;
                b_min = 0;
                b_max = 2;
            }
            else 
            {
                a_min = 0;
                a_max = 3;
                b_min = 0;
                b_max = 3;
            }
            for (int a = a_min; a < a_max; ++a)
            {
                for (int b = b_min; b < b_max; ++b)
                {
                    sob_val_x += sobel_x[a][b] * new_image_mat[i - 1 + a][j - 1 + b];
                    sob_val_y += sobel_y[a][b] * new_image_mat[i - 1 + a][j - 1 + b];
                }
                sob_val = sqrt(sob_val_x * sob_val_x + sob_val_y * sob_val_y);
                
                if (sob_val < 0)
                {
                    sob_val = 0;
                }
                if (sob_val > 255)
                {
                    sob_val = 255;
                }
                sobel_image_mat[i - grid_x][j - grid_y] = sob_val;
            }
        }
    }


    FILE *histeql;
    histeql = fopen("histeql.pgm", "w");
    fprintf(histeql, "P2\n%ld %ld\n%d\n", wid, hei, 255);

    for (int i = 0; i < hei; ++i)
    {
        for (int j = 0; j < wid; ++j)
        {
            if (j % 10 == 0)
                fprintf(histeql, "%u\n", new_image_mat[i][j]);
            else
                fprintf(histeql, "%u ", new_image_mat[i][j]);
        }
    }
    fclose(histeql);

    FILE *final;
    final = fopen("final.pgm", "w");
    fprintf(final, "P2\n%ld %ld\n%d\n", wid, hei, 255);

    for (int i = 0; i < hei; ++i)
    {
        for (int j = 0; j < wid; ++j)
        {
            if (j % 10 == 0)
                fprintf(final, "%u\n", sobel_image_mat[i][j]);
            else
                fprintf(final, "%u ", sobel_image_mat[i][j]);
        }
    }
    fclose(final);
    return 0;    
}

int readpng_init(FILE *infile, long *pWidth, long *pHeight)
{
    uch sig[8];
  
    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */
    
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 2;
    }
    
    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, NULL, NULL, NULL);
    *pWidth = width;
    *pHeight = height;

    return 0;
}

uch *readpng_get_image(double display_exponent, int *pChannels, ulg *pRowbytes)
{
    double  gamma;
    png_uint_32  i, rowbytes;
    png_bytepp  row_pointers = NULL;


    /* setjmp() must be called in every function that calls a PNG-reading
     * libpng function */

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return NULL;
    }


    /* expand palette images to RGB, low-bit-depth grayscale images to 8 bits,
     * transparency chunks to full alpha channel; strip 16-bit-per-sample
     * images to 8 bits per sample; and convert grayscale to RGB[A] */

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_expand(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_expand(png_ptr);
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);


    /* unlike the example in the libpng documentation, we have *no* idea where
     * this file may have come from--so if it doesn't have a file gamma, don't
     * do any correction ("do no harm") */

    if (png_get_gAMA(png_ptr, info_ptr, &gamma))
        png_set_gamma(png_ptr, display_exponent, gamma);


    /* all transformations have been registered; now update info_ptr data,
     * get rowbytes and channels, and allocate image memory */

    png_read_update_info(png_ptr, info_ptr);

    *pRowbytes = rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *pChannels = (int)png_get_channels(png_ptr, info_ptr);

    if ((image_data = (uch *)malloc(rowbytes*height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return NULL;
    }
    if ((row_pointers = (png_bytepp)malloc(height*sizeof(png_bytep))) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        free(image_data);
        image_data = NULL;
        return NULL;
    }

    Trace((stderr, "readpng_get_image:  channels = %d, rowbytes = %ld, height = %ld\n", *pChannels, rowbytes, height));


    /* set the individual row_pointers to point at the correct offsets */

    for (i = 0;  i < height;  ++i)
        row_pointers[i] = image_data + i*rowbytes;


    /* now we can go ahead and just read the whole image */

    png_read_image(png_ptr, row_pointers);


    /* and we're done!  (png_read_end() can be omitted if no processing of
     * post-IDAT text/time/etc. is desired) */

    free(row_pointers);
    row_pointers = NULL;

    png_read_end(png_ptr, NULL);

    return image_data;
}

void readpng_cleanup(int free_image_data)
{
    if (free_image_data && image_data) {
        free(image_data);
        image_data = NULL;
    }

    if (png_ptr && info_ptr) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        png_ptr = NULL;
        info_ptr = NULL;
    }
}