//
//  sac_model_2_orthogonal_planes.hpp
//  myvisualizer
//
//  Created by Mario Lietz on 19.03.13.
//
//
#ifndef PCL_SAMPLE_CONSENSUS_IMPL_SAC_2_ORTHOGONAL_PLANES_H_
#define PCL_SAMPLE_CONSENSUS_IMPL_SAC_2_ORTHOGONAL_PLANES_H_

#include <pcl/sample_consensus/eigen.h>
#include <pcl/sample_consensus/sac_model_2_orthogonal_planes.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/common/eigen.h>
#include <pcl/common/point_tests.h>
#include <pcl/common/angles.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> bool
pcl::SampleConsensusModelTwoOrthogonalPlanes<PointT, PointNT>::isSampleGood (const std::vector<int> &samples) const
{
	// sample normals must not be nan
	if (!pcl::isFinite (normals_->points[samples.at(0)]) ||
		!pcl::isFinite (normals_->points[samples.at(1)]))
		{
		return (false);
		}
	return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> bool
pcl::SampleConsensusModelTwoOrthogonalPlanes<PointT, PointNT>::computeModelCoefficients (
																			const std::vector<int> &samples, Eigen::VectorXf &model_coefficients)
{
	// Need 2 samples
	if (samples.size () != 2)
		{
		PCL_ERROR ("[pcl::SampleConsensusModelTwoOrthogonalPlanes::computeModelCoefficients] Invalid set of samples given (%zu)!\n", samples.size ());
		return (false);
		}
	
	if (!normals_)
		{
		PCL_ERROR ("[pcl::SampleConsensusModelTwoOrthogonalPlanes::computeModelCoefficients] No input dataset containing normals was given!\n");
		return (false);
		}
	
	std::vector<Eigen::Vector4f> normalSamples (3);
	normalSamples[0] = Eigen::Vector4f (normals_->points[samples.at(0)].normal[0], normals_->points[samples.at(0)].normal[1], normals_->points[samples.at(0)].normal[2], 0);
	normalSamples[1] = Eigen::Vector4f (normals_->points[samples.at(1)].normal[0], normals_->points[samples.at(1)].normal[1], normals_->points[samples.at(1)].normal[2], 0);
	normalSamples[2] = Eigen::Vector4f (0, 0, 0, 0);

	std::vector<Eigen::Vector3f> normalSamples3 (2);
	normalSamples3[0] = Eigen::Vector3f (normals_->points[samples.at(0)].normal[0], normals_->points[samples.at(0)].normal[1], normals_->points[samples.at(0)].normal[2]);
	normalSamples3[1] = Eigen::Vector3f (normals_->points[samples.at(1)].normal[0], normals_->points[samples.at(1)].normal[1], normals_->points[samples.at(1)].normal[2]);

	float d_normal = fabs ( (getAngle3D (normalSamples[0], normalSamples[1]))- static_cast<float>(M_PI)/2.f) ;

	// normals should be a little bit orthogonal 
	if (d_normal > eps_angle_ )
		{
		return (false);
		}

	//// calculate closest rotation as in: New Method for Extracting the Quaternion from a Rotation Matrix, Itzhack Y. Bar-Itzhack
	//// Read More: http://arc.aiaa.org/doi/abs/10.2514/2.4654
	Eigen::Matrix4f K;
	K(0,0) = (normalSamples[0](0) - normalSamples[1](1) - normalSamples[2](2))/3.;
	K(0,1) = K(1,0) = (normalSamples[0](1) + normalSamples[1](0))/3.;
	K(0,2) = K(2,0) = (normalSamples[0](2) + normalSamples[2](0))/3.;
	K(0,3) = K(3,0) = (normalSamples[1](2) - normalSamples[2](1))/3.;
	K(1,1) = (normalSamples[1](1) - normalSamples[0](0) - normalSamples[2](2))/3.;
	K(1,2) = K(2,1) = (normalSamples[1](2) + normalSamples[2](1))/3.;
	K(1,3) = K(3,1) = (normalSamples[2](0) - normalSamples[0](2))/3.;
	K(2,2) = (normalSamples[2](2) - normalSamples[0](0) - normalSamples[1](1))/3.;
	K(2,3) = K(3,2) = (normalSamples[0](1) - normalSamples[1](0))/3.;
	K(3,3) = (normalSamples[0](0) + normalSamples[1](1) + normalSamples[2](2))/3.;
	/*K(0,0) = (normalSamples[0](0) - normalSamples[1](1))/2.;
	K(0,1) = K(1,0) = (normalSamples[0](1) + normalSamples[1](0))/2.;
	K(0,2) = K(2,0) = (normalSamples[0](2))/2.;
	K(0,3) = K(3,0) = (- normalSamples[2](1))/2.;
	K(1,1) = (normalSamples[1](1) - normalSamples[0](0))/2.;
	K(1,2) = K(2,1) = (normalSamples[2](1))/2.;
	K(1,3) = K(3,1) = (normalSamples[2](0))/2.;
	K(2,2) = (- normalSamples[0](0) - normalSamples[1](1))/2.;
	K(2,3) = K(3,2) = (normalSamples[0](1) - normalSamples[1](0))/2.;
	K(3,3) = (normalSamples[0](0) + normalSamples[1](1))/2.;
	*/
	// take eigenvector corresponding to largest eigenvalue
	Eigen::EigenSolver<Eigen::Matrix4f> es (K);
	Eigen::Vector4cf Eigenvalues = es.eigenvalues();
	float maxEV = abs(Eigenvalues(0));
	int maxEV_Ind = 0;
	for (int i = 1; i < 4 ; i++)
		{
		if (abs(Eigenvalues(i)) > maxEV)
			{
			maxEV = abs(Eigenvalues(i));
			maxEV_Ind = i;
			}
		}
	// set rotation as quaternion
	Eigen::Vector4f Eigenvector = es.eigenvectors ().col (maxEV_Ind).real ();
	Eigen::Quaternionf rotationQuaternion (Eigenvector);
	rotationQuaternion.normalize ();

	std::vector<Eigen::Vector4f> planes (2);
	planes[0] = Eigen::Vector4f (normalSamples[0].coeff(0), normalSamples[0].coeff(1), normalSamples[0].coeff(2),
		-(normalSamples[0].coeff(0)*input_->points[samples[0]].x + 
		normalSamples[0].coeff(1)*input_->points[samples[0]].y +
		normalSamples[0].coeff(2)*input_->points[samples[0]].z ) );
	planes[1] = Eigen::Vector4f (normalSamples[1].coeff(0), normalSamples[1].coeff(1), normalSamples[1].coeff(2),
		-(normalSamples[1].coeff(0)*input_->points[samples[1]].x + 
		normalSamples[1].coeff(1)*input_->points[samples[1]].y +
		normalSamples[1].coeff(2)*input_->points[samples[1]].z ) );

	Eigen::VectorXf intersectionLine;

	planeWithPlaneIntersection (planes[0], planes[1], intersectionLine);

	Eigen::Matrix3f rotationMatrix = rotationQuaternion.toRotationMatrix ();

	model_coefficients.resize (9);
	// first 3 values of intersectionLine should be one point on the line
	model_coefficients[0] = intersectionLine[0];
	model_coefficients[1] = intersectionLine[1];
	model_coefficients[2] = intersectionLine[2];
	model_coefficients[3] = rotationMatrix(0,0);
	model_coefficients[4] = rotationMatrix(1,0);
	model_coefficients[5] = rotationMatrix(2,0);
	model_coefficients[6] = rotationMatrix(0,1);
	model_coefficients[7] = rotationMatrix(1,1);
	model_coefficients[8] = rotationMatrix(2,1);

	//model_coefficients[3] = rotationQuaternion.x ();
	//model_coefficients[4] = rotationQuaternion.y ();
	//model_coefficients[5] = rotationQuaternion.z ();
	//model_coefficients[6] = rotationQuaternion.w ();
	
	std::vector<Eigen::Vector4f> normalVectors (3);
	for (int i = 0; i < 3; i++)
		{
		normalVectors.at(i) = Eigen::Vector4f (rotationMatrix(0,i), rotationMatrix(1,i), rotationMatrix(2,i), 0.f);
		}
	std::vector<float> distanceNormal (6);
	
	
	// Debugging: Find corresponding normals of model and samples
	for (int i = 0; i < 2; i++)
		{
		distanceNormal.at (i*3) =  fabs(getAngle3D (normalVectors.at(0), normalSamples[i]) - static_cast<float>(M_PI)/2.f) ;
		distanceNormal.at (i*3 + 1) = fabs(getAngle3D (normalVectors.at(1), normalSamples[i]) - static_cast<float>(M_PI)/2.f) ;
		distanceNormal.at (i*3 + 2) = fabs(getAngle3D (normalVectors.at(2), normalSamples[i]) - static_cast<float>(M_PI)/2.f) ;
		}
	//best elements should be roughly pi/2
	std::vector<float>::iterator minDistance = std::max_element(distanceNormal.begin(), distanceNormal.end());  
    int minDistanceInd = std::distance(distanceNormal.begin(), minDistance);
	// corresponding elements
	std::vector<int> sampleInd (2), modelInd (3);
	sampleInd[0] = minDistanceInd / 3;
	modelInd[0] = minDistanceInd % 3;
	
	int Il[3] = {0, 1, 2};
	std::vector<int> modelIndices (Il,Il+3),  sampleIndices (Il,Il+2);
	modelIndices.erase (modelIndices.begin() + modelInd[0]);
	sampleIndices.erase (sampleIndices.begin() + sampleInd[0]);
	
	distanceNormal.erase(distanceNormal.begin () + sampleInd[0]*3, distanceNormal.begin () + sampleInd[0]*3 +3);
	distanceNormal.erase(distanceNormal.begin () + modelInd[0]);
	

	minDistance = std::max_element(distanceNormal.begin(), distanceNormal.end());
    minDistanceInd = std::distance(distanceNormal.begin(), minDistance);
	sampleInd[1] = sampleIndices.at(minDistanceInd / 2);
	modelInd[1]  = modelIndices.at(minDistanceInd % 2);
	
	modelIndices.erase (modelIndices.begin() + minDistanceInd / 2);
	sampleIndices.erase (sampleIndices.begin() + minDistanceInd % 2);
	
	if (modelInd[0] != sampleInd[0] || modelInd[1] != sampleInd[1] )
		{
		PCL_ERROR ("[pcl::SampleConsensusModelTwoOrthogonalPlaness::sac_model_2_ortho_planes] Wrong order of sample to modelnormals: so still some coding necassary!!!!\n");
		}
	return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelTwoOrthogonalPlanes<PointT, PointNT>::getDistancesToModel (
																	   const Eigen::VectorXf &model_coefficients, std::vector<double> &distances)
{
	// Check if the model is valid given the user constraints
	if (!isModelValid (model_coefficients))
		{
		distances.clear ();
		return;
		}	
	distances.resize (indices_->size ());
	
	//rotationQuaternion.normalize ();
	
	std::vector<Eigen::Vector4f> planeCoefficients(2), planeNormals(2);
	
	planeCoefficients.at(0) = Eigen::Vector4f (model_coefficients[3], model_coefficients[4], model_coefficients[5], -(model_coefficients[3]*model_coefficients[0] + model_coefficients[4]*model_coefficients[1] + model_coefficients[5]*model_coefficients[2]) ) ;
	planeCoefficients.at(1) = Eigen::Vector4f (model_coefficients[6], model_coefficients[7], model_coefficients[8], -(model_coefficients[6]*model_coefficients[0] + model_coefficients[7]*model_coefficients[1] + model_coefficients[8]*model_coefficients[2]) ) ;

	planeNormals.at(0) = Eigen::Vector4f(model_coefficients[3], model_coefficients[4], model_coefficients[5], 0);
	planeNormals.at(1) = Eigen::Vector4f(model_coefficients[6], model_coefficients[7], model_coefficients[8], 0);
	
	// the model coefficients (Tx, Ty, Tz, Qx, Qy, Qz, Qw)
	for (size_t i = 0 ; i < indices_->size () ; i++ )
		{
		Eigen::Vector4f pt (input_->points[(*indices_)[i]].x, input_->points[(*indices_)[i]].y, input_->points[(*indices_)[i]].z, 1);
		const PointNT &nt = normals_->points[(*indices_)[i]];
		Eigen::Vector4f n (nt.normal_x, nt.normal_y, nt.normal_z, 0);
		// Weight with the point curvature. On flat surfaces, curvature -> 0, which means the normal will have a higher influence
		// curvature isn't avaible on all normal methods
		double weight = normal_distance_weight_;
		if (!pcl_isnan (nt.curvature))
			{
			weight = normal_distance_weight_ * (1.0 - nt.curvature);
			}

		std::vector<double> distance(2);
		
		// Calculate the angular distance between the point normal and the plane normal
		double d_normal = fabs(getAngle3D(n, planeNormals[0]));
		d_normal = (std::min) (d_normal, M_PI - d_normal);
		distance.at(0) = fabs (weight * d_normal + (1.0 - weight) * fabs (planeCoefficients.at(0).dot (pt) ));

		d_normal = fabs(getAngle3D(n, planeNormals[1]));
		d_normal = (std::min) (d_normal, M_PI - d_normal);
		distance.at(1) = fabs (weight * d_normal + (1.0 - weight) * fabs (planeCoefficients.at(1).dot (pt) ));

				
		distances[i] = std::min (distance[0], distance[1]); 
		}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelTwoOrthogonalPlanes<PointT, PointNT>::selectWithinDistance (
																		const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers)
{
	// Check if the model is valid given the user constraints
	if (!isModelValid (model_coefficients))
		{
		inliers.clear ();
		return;
		}
	
	int nr_p = 0;
	inliers.resize (indices_->size ());
	error_sqr_dists_.resize (indices_->size ());
	std::vector<double> distances (indices_->size ());
	getDistancesToModel (model_coefficients, distances);
	// Iterate through the 3d points and calculate the distances from them to the sphere
	for (size_t i = 0; i < indices_->size (); ++i)
		{
		if (distances[i] < threshold)
			{
			// Returns the indices of the points whose distances are smaller than the threshold
			inliers[nr_p] = (*indices_)[i];
			error_sqr_dists_[nr_p] = distances[i];
			++nr_p;
			}
		}
	inliers.resize (nr_p);
	error_sqr_dists_.resize (nr_p);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> int
pcl::SampleConsensusModelTwoOrthogonalPlanes<PointT, PointNT>::countWithinDistance (
																	   const Eigen::VectorXf &model_coefficients, const double threshold)
{
	// Check if the model is valid given the user constraints
	if (!isModelValid (model_coefficients))
		return (0);
	
	int nr_p = 0;
	std::vector<double> distances (indices_->size ());
	getDistancesToModel (model_coefficients, distances);
	
	// Iterate through the 3d points and calculate the distances from them to the sphere
	for (size_t i = 0; i < indices_->size (); ++i)
		{
		if (distances[i] < threshold)
			{
			// Returns the indices of the points whose distances are smaller than the threshold
			++nr_p;
			}
		}
	return (nr_p);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelTwoOrthogonalPlanes<PointT, PointNT>::optimizeModelCoefficients (
																			 const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients, Eigen::VectorXf &optimized_coefficients)
{
	optimized_coefficients = model_coefficients;
	
	// Needs a set of valid model coefficients
	if (model_coefficients.size () != 9)
		{
		PCL_ERROR ("[pcl::SampleConsensusModelTwoOrthogonalPlanes::optimizeModelCoefficients] Invalid number of model coefficients given (%zu)!\n", model_coefficients.size ());
		return;
		}
	
	if (inliers.empty ())
		{
		PCL_DEBUG ("[pcl::SampleConsensusModelTwoOrthogonalPlanes:optimizeModelCoefficients] Inliers vector empty! Returning the same coefficients.\n");
		return;
		}

	tmp_inliers_ = &inliers;

	// optimization must respect the orthogonalaty condition: using quaterion should fulfill this requirement
	Eigen::Matrix3f rotationMatrix;
	rotationMatrix(0,0) = model_coefficients[3];
	rotationMatrix(1,0) = model_coefficients[4];
	rotationMatrix(2,0) = model_coefficients[5];
	rotationMatrix(0,1) = model_coefficients[6];
	rotationMatrix(1,1) = model_coefficients[7];
	rotationMatrix(2,1) = model_coefficients[8];
	rotationMatrix(0,2) = 0;
	rotationMatrix(1,2) = 0;
	rotationMatrix(2,2) = 0;
	rotationMatrix.transposeInPlace ();

	Eigen::Matrix4f K;
	K(0,0) = (rotationMatrix(0,0) - rotationMatrix(1,1) - rotationMatrix(2,2))/3.;
	K(0,1) = K(1,0) = (rotationMatrix(0,1) + rotationMatrix(1,0))/3.;
	K(0,2) = K(2,0) = (rotationMatrix(0,2) + rotationMatrix(2,0))/3.;
	K(0,3) = K(3,0) = (rotationMatrix(1,2) - rotationMatrix(2,1))/3.;
	K(1,1) = (rotationMatrix(1,1) - rotationMatrix(0,0) - rotationMatrix(2,2))/3.;
	K(1,2) = K(2,1) = (rotationMatrix(1,2) + rotationMatrix(2,1))/3.;
	K(1,3) = K(3,1) = (rotationMatrix(2,0) - rotationMatrix(0,2))/3.;
	K(2,2) = (rotationMatrix(2,2) - rotationMatrix(0,0) - rotationMatrix(1,1))/3.;
	K(2,3) = K(3,2) = (rotationMatrix(0,1) - rotationMatrix(1,0))/3.;
	K(3,3) = (rotationMatrix(0,0) + rotationMatrix(1,1) + rotationMatrix(2,2))/3.;

	// take eigenvector corresponding to largest eigenvalue
	Eigen::EigenSolver<Eigen::Matrix4f> es (K);
	Eigen::Vector4cf Eigenvalues = es.eigenvalues();
	float maxEV = abs(Eigenvalues(0));
	int maxEV_Ind = 0;
	for (int i = 1; i < 4 ; i++)
		{
		if (abs(Eigenvalues(i)) > maxEV)
			{
			maxEV = abs(Eigenvalues(i));
			maxEV_Ind = i;
			}
		}
	// set rotation as quaternion
	Eigen::Vector4f Eigenvector = es.eigenvectors ().col (maxEV_Ind).real ();

	Eigen::VectorXf coefficientsQuaternion;
	coefficientsQuaternion.resize (7);
	coefficientsQuaternion[0] = model_coefficients[0];
	coefficientsQuaternion[1] = model_coefficients[1];
	coefficientsQuaternion[2] = model_coefficients[2];
	coefficientsQuaternion[3] = Eigenvector[0];
	coefficientsQuaternion[4] = Eigenvector[1];
	coefficientsQuaternion[5] = Eigenvector[2];
	coefficientsQuaternion[6] = Eigenvector[3];

	OptimizationFunctor functor (static_cast<int> (inliers.size ()), this);
	Eigen::NumericalDiff<OptimizationFunctor > num_diff (functor);
	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<OptimizationFunctor>, float> lm (num_diff);
	int info = lm.minimize (coefficientsQuaternion);

	Eigen::Quaternionf rotationQuaternion (coefficientsQuaternion[6], coefficientsQuaternion[3], coefficientsQuaternion[4], coefficientsQuaternion[5]);
	rotationQuaternion.normalize ();
	rotationMatrix = rotationQuaternion.toRotationMatrix ();

	optimized_coefficients[0] = coefficientsQuaternion[0];
	optimized_coefficients[1] = coefficientsQuaternion[1];
	optimized_coefficients[2] = coefficientsQuaternion[2];
	optimized_coefficients[3] = rotationMatrix(0,0);
	optimized_coefficients[4] = rotationMatrix(1,0);
	optimized_coefficients[5] = rotationMatrix(2,0);
	optimized_coefficients[6] = rotationMatrix(0,1);
	optimized_coefficients[7] = rotationMatrix(1,1);
	optimized_coefficients[8] = rotationMatrix(2,1);

	
	// Compute the L2 norm of the residuals
	PCL_DEBUG ("[pcl::SampleConsensusModelCylinder::optimizeModelCoefficients] LM solver finished with exit code %i, having a residual norm of %g. \nInitial solution: %g %g %g %g %g %g %g \nFinal solution: %g %g %g %g %g %g %g\n",
				info, lm.fvec.norm (), model_coefficients[0], model_coefficients[1], model_coefficients[2], model_coefficients[3],
				model_coefficients[4], model_coefficients[5], model_coefficients[6], model_coefficients[7], model_coefficients[8], optimized_coefficients[0], optimized_coefficients[1], optimized_coefficients[2], optimized_coefficients[3], optimized_coefficients[4], optimized_coefficients[5], optimized_coefficients[6], optimized_coefficients[7], optimized_coefficients[8]);
	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelTwoOrthogonalPlanes<PointT, PointNT>::projectPoints (
																 const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients, PointCloud &projected_points, bool copy_data_fields)
{
	// Needs a valid set of model coefficients
	if (model_coefficients.size () != 9)
		{
		PCL_ERROR ("[pcl::SampleConsensusModelTwoOrthogonalPlanes::projectPoints] Invalid number of model coefficients given (%zu)!\n", model_coefficients.size ());
		return;
		}
	
	PCL_ERROR ("[pcl::SampleConsensusModelTwoOrthogonalPlanes::projectPoints] not implemented yet!\n", model_coefficients.size ());
	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> bool
pcl::SampleConsensusModelTwoOrthogonalPlanes<PointT, PointNT>::doSamplesVerifyModel (
																		const std::set<int> &indices, const Eigen::VectorXf &model_coefficients, const double threshold)
{
	// Needs a valid model coefficients
	if (model_coefficients.size () != 9)
		{
		PCL_ERROR ("[pcl::SampleConsensusModelTwoOrthogonalPlanes::doSamplesVerifyModel] Invalid number of model coefficients given (%zu)!\n", model_coefficients.size ());
		return (false);
		}
	std::vector<double> distances (indices.size ());
	getDistancesToModel(model_coefficients, distances);
	for (std::set<int>::const_iterator it = indices.begin (); it != indices.end (); ++it)
		{
		// Aproximate the distance from the point to the cylinder as the difference between
		// dist(point,cylinder_axis) and cylinder radius
		// @note need to revise this.
		if (distances[*it] > threshold)
			return (false);
		}
	
	return (true);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////




#define PCL_INSTANTIATE_SampleConsensusModelTwoOrthogonalPlanes(PointT, PointNT)	template class PCL_EXPORTS pcl::SampleConsensusModelTwoOrthogonalPlanes<PointT, PointNT>;

#endif //PCL_SAMPLE_CONSENSUS_IMPL_SAC_2_ORTHOGONAL_PLANES_H_