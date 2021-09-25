
#include "../../cpp_utils/cloud/cloud.h"

#include <set>
#include <cstdint>

using namespace std;

class SampledData
{
public:

	// Elements
	// ********

	int count;
	PointXYZ point;
	vector<float> features;
	vector<unordered_map<int, int>> semantic_labels;
	vector<unordered_map<int, int>> instance_labels;


	// Methods
	// *******

	// Constructor
	SampledData() 
	{ 
		count = 0; 
		point = PointXYZ();
	}

	SampledData(const size_t fdim, const size_t ldim)
	{
		count = 0;
		point = PointXYZ();
	    features = vector<float>(fdim);
	    semantic_labels = vector<unordered_map<int, int>>(ldim);
		instance_labels = vector<unordered_map<int, int>>(ldim);
	}

	// Method Update
	void update_all(const PointXYZ p, vector<float>::iterator f_begin, vector<int>::iterator s_l_begin, vector<int>::iterator i_l_begin)
	{
		count += 1;
		point += p;
		transform (features.begin(), features.end(), f_begin, features.begin(), plus<float>());
		int i = 0;
		for(vector<int>::iterator it = s_l_begin; it != s_l_begin + semantic_labels.size(); ++it)
		{
		    semantic_labels[i][*it] += 1;
		    i++;
		}
		i = 0;
		for(vector<int>::iterator it = i_l_begin; it != i_l_begin + instance_labels.size(); ++it)
		{
		    instance_labels[i][*it] += 1;
		    i++;
		}
		return;
	}
	void update_features(const PointXYZ p, vector<float>::iterator f_begin)
	{
		count += 1;
		point += p;
		transform (features.begin(), features.end(), f_begin, features.begin(), plus<float>());
		return;
	}
	void update_classes(const PointXYZ p, vector<int>::iterator s_l_begin, vector<int>::iterator i_l_begin)
	{
		count += 1;
		point += p;
		int i = 0;
		for(vector<int>::iterator it = s_l_begin; it != s_l_begin + semantic_labels.size(); ++it)
		{
		    semantic_labels[i][*it] += 1;
		    i++;
		}
		i = 0;
		for(vector<int>::iterator it = i_l_begin; it != i_l_begin + instance_labels.size(); ++it)
		{
		    instance_labels[i][*it] += 1;
		    i++;
		}
		return;
	}
	void update_points(const PointXYZ p)
	{
		count += 1;
		point += p;
		return;
	}
};

void grid_subsampling(vector<PointXYZ>& original_points,
                      vector<PointXYZ>& subsampled_points,
                      vector<float>& original_features,
                      vector<float>& subsampled_features,
                      vector<int>& original_semantic_classes,
					  vector<int>& original_instance_classes,
                      vector<int>& subsampled_semantic_classes,
                      vector<int>& subsampled_instance_classes,
                      float sampleDl,
                      int verbose);

void batch_grid_subsampling(vector<PointXYZ>& original_points,
                            vector<PointXYZ>& subsampled_points,
                            vector<float>& original_features,
                            vector<float>& subsampled_features,
                            vector<int>& original_semantic_classes,
							vector<int>& original_instance_classes,
							vector<int>& subsampled_semantic_classes,
							vector<int>& subsampled_instance_classes,
                            vector<int>& original_batches,
                            vector<int>& subsampled_batches,
                            float sampleDl,
                            int max_p);

