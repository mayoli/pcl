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
	/** \brief Set the angle epsilon (delta) threshold.
    * \param[in] ea the maximum allowed difference between orthogonality and the angle between 2 normals.
    */
    inline void 
    setEpsAngle (const double ea) { eps_angle_ = ea; }

    /** \brief Get the angle epsilon (delta) threshold. */
    inline double 
    getEpsAngle () { return (eps_angle_); }


	/** \brief Check whether the given index samples can form a valid model, compute the model coefficients from
	 * these samples and store them internally in model_coefficients_. The coefficients are:
	 * center.x
	 * center.y
	 * center.z
	 * Quaternion.x
	 * Quaternion.y
	 * Quaternion.z
	 * Quaternion.w
	 * \param[in] samples the point indices found as possible good candidates for creating a valid model
	 * \param[out] model_coefficients the resultant model coefficients
	 */
	bool
	computeModelCoefficients (const std::vector<int> &samples,
							  Eigen::VectorXf &model_coefficients);
	
	/** \brief Compute all distances from the cloud data to the given 3 orthogonal planes.
	 * \param[in] model_coefficients the coefficients of a 3 orthogonal planes model that we need to compute distances to
	 * \param[out] distances the resultant estimated distances
	 */
	void
	getDistancesToModel (const Eigen::VectorXf &model_coefficients,
						 std::vector<double> &distances);
	
	/** \brief Select all the points which respect the given model coefficients as inliers.
	 * \param[in] model_coefficients the coefficients of a 3 orthogonal planes model that we need to compute distances to
	 * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
	 * \param[out] inliers the resultant model inliers
	 */
	void
	selectWithinDistance (const Eigen::VectorXf &model_coefficients,
						  const double threshold,
						  std::vector<int> &inliers);
	
	/** \brief Count all the points which respect the given model coefficients as inliers.
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
	
	/** \brief Verify whether a subset of indices verifies the given 3 orthogonal planes model coefficients.
	 * \param[in] indices the data indices that need to be tested against the 3 orthogonal planes model
	 * \param[in] model_coefficients the 3 orthogonal planes model coefficients
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
	/** \brief maximum allowed difference between orthogonality and the angle between 2 normals. */
	double eps_angle_;


	
	/** \brief Check if a sample of indices results in a good sample of points
	 * indices.
	 * \param[in] samples the resultant index samples
	 */
	virtual bool
	isSampleGood (const std::vector<int> &samples) const;


	
	/** \brief temporary pointer to a list of given indices for optimizeModelCoefficients () */
	const std::vector<int> *tmp_inliers_;
	

#if defined BUILD_Maintainer && defined __GNUC__ && __GNUC__ == 4 && __GNUC_MINOR__ > 3
#pragma GCC diagnostic ignored "-Weffc++"
#endif
	/** \brief Functor for the optimization function */
	struct OptimizationFunctor : pcl::Functor<float>
	{
	/** Functor constructor
		* \param[in] m_data_points the number of data points to evaluate
		* \param[in] estimator pointer to the estimator object
		* \param[in] distance distance computation function pointer
		*/
	OptimizationFunctor (int m_data_points, pcl::SampleConsensusModelThreeOrthogonalPlanes<PointT, PointNT> *model) : 
		pcl::Functor<float> (m_data_points), model_ (model) {}

	/** Cost function to be minimized
		* \param[in] x variables array
		* \param[out] fvec resultant functions evaluations
		* \return 0
		*/
	int 
	operator() (const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
	{
	Eigen::Quaternionf rotationQuaternion (x[6],x[3],x[4],x[5]);
	rotationQuaternion.normalize ();
	Eigen::Matrix3f rotationMatrix = rotationQuaternion.toRotationMatrix ();
	
	std::vector<Eigen::Vector4f> planeCoefficients (3);
	
	planeCoefficients.at(0) = (Eigen::Vector4f (rotationMatrix.coeff(0,0),rotationMatrix.coeff(1,0),rotationMatrix.coeff(2,0), -(rotationMatrix.coeff(0,0)*x[0] + rotationMatrix.coeff(1,0)*x[1] + rotationMatrix.coeff(2,0)*x[2]) ) );
	planeCoefficients.at(1) = (Eigen::Vector4f (rotationMatrix.coeff(0,1),rotationMatrix.coeff(1,1),rotationMatrix.coeff(2,1), -(rotationMatrix.coeff(0,1)*x[0] + rotationMatrix.coeff(1,1)*x[1] + rotationMatrix.coeff(2,1)*x[2]) ) );
	planeCoefficients.at(2) = (Eigen::Vector4f (rotationMatrix.coeff(0,2),rotationMatrix.coeff(1,2),rotationMatrix.coeff(2,2), -(rotationMatrix.coeff(0,2)*x[0] + rotationMatrix.coeff(1,2)*x[1] + rotationMatrix.coeff(2,2)*x[2]) ) );
	// the model coefficients (Tx, Ty, Tz, Qx, Qy, Qz, Qw)
	
	for  (int i = 0; i < values (); ++i)
		{
		Eigen::Vector4f pt (model_->input_->points[(*model_->tmp_inliers_)[i]].x, model_->input_->points[(*model_->tmp_inliers_)[i]].y, model_->input_->points[(*model_->tmp_inliers_)[i]].z, 1);
		std::vector<float> distance(3);
		
		// Calculate the distance between the point normal and the plane normal
		distance.at(0) = fabs(planeCoefficients.at(0).dot (pt));
		distance.at(1) = fabs(planeCoefficients.at(1).dot (pt));
		distance.at(2) = fabs(planeCoefficients.at(2).dot (pt));
				
		fvec[i] = *std::min_element(distance.begin(), distance.end()); 
		//fvec[i] = pow(static_cast<float>(*std::min_element(distance.begin(), distance.end())) ,2 ); 
		}

		return (0);
	}

	pcl::SampleConsensusModelThreeOrthogonalPlanes<PointT, PointNT> *model_;
	};
#if defined BUILD_Maintainer && defined __GNUC__ && __GNUC__ == 4 && __GNUC_MINOR__ > 3
#pragma GCC diagnostic warning "-Weffc++"
#endif

	};
}
#ifdef PCL_NO_PRECOMPILE
#include <pcl/segmentation/impl/sac_model_3_orthogonal_planes.hpp>
#endif

#endif  //#ifndef PCL_SAMPLE_CONSENSUS_MODEL_3_ORTHOGONAL_PLANES_H_



