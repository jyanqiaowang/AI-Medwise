Script for illumination quality analysis of videos.

Computes average frame brightness for each video.

Applies three classification methods:

Fixed threshold (default = 50).

Dataset-driven median brightness.

Dataset-driven percentile brightness (e.g., 25th percentile).

Classifies videos into Poor, Uncertain, or Good based on the fraction of frames above threshold.

Generates histograms and structured CSV outputs.