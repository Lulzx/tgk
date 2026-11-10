/* Dump TGK geometry coefficients for an input file. */
#include "../tgk.h"
#include <stdlib.h>
int main(int argc, char **argv) {
    tgk_param p; tgk_param_defaults(&p);
    if (argc > 1 && tgk_param_parse_file(&p, argv[1])) return 1;
    tgk_geom g; tgk_geom_init(&p, &g);
    printf("# z bmag dbdz gradpar gxx gxy gyy cv_x cv_y gb_x gb_y jac\n");
    for (int i = 0; i < p.nz; i++)
        printf("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", g.z[i], g.bmag[i], g.dbdz[i],
               g.gradpar[i], g.gxx[i], g.gxy[i], g.gyy[i], g.cv_x[i], g.cv_y[i], g.gb_x[i], g.gb_y[i], g.jac[i]);
    return 0;
}
