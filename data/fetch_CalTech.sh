wget 'https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_16_split1.mat'
wget 'https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat'
mkdir -p 'CalTech_101_Silhouettes'
mv caltech101_silhouettes_16_split1.mat CalTech_101_Silhouettes
mv caltech101_silhouettes_28_split1.mat CalTech_101_Silhouettes
rm -f ./wget-log
