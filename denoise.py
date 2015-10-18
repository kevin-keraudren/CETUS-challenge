#!/usr/bin/python

import irtk
import sys
import scipy.ndimage as nd

input_file = sys.argv[1]
output_file = sys.argv[2]

img = irtk.imread( input_file, dtype='float32' )

img = irtk.Image( nd.median_filter(img.get_data(), 5),
                  img.get_header() )
irtk.imwrite(output_file, img )
