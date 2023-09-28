# light clustering

Our paper<sup>[1]</sup> utilizes the Gonzalez algorithm<sup>[2]</sup>. 

The Gonzalez algorithm for k center is a greedy algorithm that aims to find k centers in a set of data points such that the maximum distance from any data point to its nearest center is minimized. The algorithm works as follows:

- Choose an arbitrary data point as the first center.
- Repeat k-1 times:
  - Find the data point that is farthest from its nearest center and add it as a new center.
- Return the k centers.

[1] Is Simple Uniform Sampling Effective for Center-Based Clustering with Outliers: When and Why?

[2] Teofilo F Gonzalez. Clustering to minimize the maximum intercluster distance. *Theoretical Computer Science*, 38:293– 306, 1985.