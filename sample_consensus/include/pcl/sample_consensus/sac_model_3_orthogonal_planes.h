//
//  sac_model_3_orthogonal_planes.h
//  myvisualizer
//
//  Created by Mario Lietz on 10.03.13.
//
//

#ifndef PCL_SAMPLE_CONSENSUS_MODEL_3_ORTHOGONAL_PLANES_H_
#define PCL_SAMPLE_CONSENSUS_MODEL_3_ORTHOGONAL_PLANES_H_


#include <pcl/sample_consensus/sac_model.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/common/eigen.h>

namespace pcl
{
	template <typename PointT, typename PointNT>
	class SampleConsensusModelThreeOrthogonalPlanes : public SampleConsensusModel<PointT> , public SampleConsensusModelFromNormals<PointT, PointNT>
	{
	public:
	using SampleConsensusModel<PointT>::input_;
	using SampleConsensusModel<PointT>::indices_;
	using SampleConsensusModelFromNormals<PointT, PointNT>::normals_;
	using SampleConsensusModelFromNormals<PointT, PointNT>::normal_distance_weight_;
	using SampleConsensusModel<PointT>::error_sqr_dists_;
	
	typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
	typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
	typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
	
	typedef boost::shared_ptr<SampleConsensusModelThreeOrthogonalPlanes> Ptr;
	
	/** \brief Constructor for base SampleConsensusModelPlane.
	 * \param[in] cloud the input point cloud dataset
	 * \param[in] random if true set the random seed to the current time, else set to 12345 (default: false)
	 */
	SampleConsensusModelThreeOrthogonalPlanes (const PointCloudConstPtr &cloud, bool random = false)
	: SampleConsensusModel<PointT> (cloud, random)
	, SampleConsensusModelFromNormals<PointT, PointNT> ()
	, eps_angle_ (0.15) {};
	
	/** \brief Constructor for base SampleConsensusModelPlane.
	 * \param[in] cloud the input point cloud dataset
	 * \param[in] indices a vector of point indices to be used from \a cloud
	 * \param[in] random if true set the random seed to the current time, else set to 12345 (default: false)
	 */
	SampleConsensusModelThreeOrthogonalPlanes (const PointCloudConstPtr &cloud,
								const std::vector<int> &indices,
								bool random = false)
	: SampleConsensusModel<PointT> (cloud, indices, random)
	, SampleConsensusModelFromNormals<PointT, PointNT> ()
	, eps_angle_ (0.15) {};
	
	/** \brief Empty destructor */
	virtual ~SampleConsensusModelThreeOrthogonalPlanes () {};
	/** \brief Copy constructor.
	 * \param[in] source the model to copy into this
	 */
	SampleConsensusModelThreeOrthogonalPlanes (const SampleConsensusModelThreeOrthogonalPlanes &source) :
    SampleConsensusModel<PointT> (), 
    SampleConsensusModelFromNormals<PointT, PointNT> (),
    eps_angle_ (0.15)
    {
    *this = source;
    }

	inline SampleConsensusModelThreeOrthogonalPlanes&
	operator = (const SampleConsensusModelThreeOrthogonalPlanes &source)
	{
	SampleConsensusModel<PointT>::operator=(source);
	eps_angle_ = source.eps_angle_;
	return (*this);
	};
	/** \brief Check whether the given index samples can form a valid plane model, compute the model coefficients from
	 * these samples and store them internally in model_coefficients_. The plane coefficients are:
	 * a, b, c, d (ax+by+cz+d=0)
	 * \param[in] samples the point indices found as possible good candidates for creating a valid model
	 * \param[out] model_coefficients the resultant model coefficients
	 */
	bool
	computeModelCoefficients (const std::vector<int> &samples,
							  Eigen::VectorXf &model_coefficients);
	
	/** \brief Compute all distances from the cloud data to a given plane model.
	 * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
	 * \param[out] distances the resultant estimated distances
	 */
	void
	getDistancesToModel (const Eigen::VectorXf &model_coefficients,
						 std::vector<double> &distances);
	
	/** \brief Select all the points which respect the given model coefficients as inliers.
	 * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
	 * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
	 * \param[out] inliers the resultant model inliers
	 */
	void
	selectWithinDistance (const Eigen::VectorXf &model_coefficients,
						  const double threshold,
						  std::vector<int> &inliers);
	
	/** \brief Count all the points which respect the given model coefficients as inliers.
	 *
	 * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
	 * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
	 * \return the resultant number of inliers
	 */
	virtual int
	countWithinDistance (const Eigen::VectorXf &model_coefficients,
						 const double threshold);
	
	/** \brief Recompute the plane coefficients using the given inlier set and return them to the user.
	 * @note: these are the coefficients of the plane model after refinement (eg. after SVD)
	 * \param[in] inliers the data inliers found as supporting the model
	 * \param[in] model_coefficients the initial guess for the model coefficients
	 * \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
	 */
	void
	optimizeModelCoefficients (const std::vector<int> &inliers,
							   const Eigen::VectorXf &model_coefficients,
							   Eigen::VectorXf &optimized_coefficients);
	
	/** \brief Create a new point cloud with inliers projected onto the plane model.
	 * \param[in] inliers the data inliers that we want to project on the plane model
	 * \param[in] model_coefficients the *normalized* coefficients of a plane model
	 * \param[out] projected_points the resultant projected points
	 * \param[in] copy_data_fields set to true if we need to copy the other data fields
	 */
	void
	projectPoints (const std::vector<int> &inliers,
				   const Eigen::VectorXf &model_coefficients,
				   PointCloud &projected_points,
				   bool copy_data_fields = true);
	
	/** \brief Verify whether a subset of indices verifies the given plane model coefficients.
	 * \param[in] indices the data indices that need to be tested against the plane model
	 * \param[in] model_coefficients the plane model coefficients
	 * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
	 */
	bool
	doSamplesVerifyModel (const std::set<int> &indices,
						  const Eigen::VectorXf &model_coefficients,
						  const double threshold);
	
	/** \brief Return an unique id for this model (SACMODEL_PLANE). */
	inline pcl::SacModel
	getModelType () const { return (SACMODEL_3_ORTHOGONAL_PLANES); }
	
	protected:
	/** \brief Check whether a model is valid given the user constraints.
	 * \param[in] model_coefficients the set of model coefficients
	 */
	inline bool
	isModelValid (const Eigen::VectorXf &model_coefficients)
	{
	// Needs a valid model coefficients
	if (model_coefficients.size () != 7)
		{
		PCL_ERROR ("[pcl::SampleConsensusModelThreeOrthogonalPlanes::isModelValid] Invalid number of model coefficients given (%zu)!\n", model_coefficients.size ());
		return (false);
		}
	return (true);
	}
	
	private:
	/** \brief The maximum allowed difference between the plane normal and the given axis. */
	double eps_angle_;
	
	/** \brief Check if a sample of indices results in a good sample of points
	 * indices.
	 * \param[in] samples the resultant index samples
	 */
	virtual bool
	isSampleGood (const std::vector<int> &samples) const;
	};
}
#ifdef PCL_NO_PRECOMPILE
#include <pcl/segmentation/impl/sac_model_3_orthogonal_planes.hpp>
#endif

#endif  //#ifndef PCL_SAMPLE_CONSENSUS_MODEL_3_ORTHOGONAL_PLANES_H_



