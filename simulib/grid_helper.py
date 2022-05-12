import numpy as np
from simulation_functions import getMapLocation, createMeshFromPoints, getElevationMap, rotate, llh2enu, genPulse
import open3d as o3d


class Environment(object):
    _mesh = None
    _pcd = None
    _ref_coefs = None
    _scat_coefs = None

    def __init__(self, pts=None, scattering=None, reflectivity=None):
        if pts is not None:
            # Create the point cloud for the mesh basis
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)

            # Downsample if possible to reduce number of triangles
            pcd = pcd.voxel_down_sample(voxel_size=np.mean(pcd.compute_nearest_neighbor_distance()) / 1.5)
            its = 0
            while np.std(pcd.compute_nearest_neighbor_distance()) > 2. and its < 30:
                dists = pcd.compute_nearest_neighbor_distance()
                pcd = pcd.voxel_down_sample(voxel_size=np.mean(dists) / 1.5)
                its += 1

            avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
            radius = 3 * avg_dist
            radii = [radius, radius * 2]
            pcd.estimate_normals()
            try:
                pcd.orient_normals_consistent_tangent_plane(100)
            except RuntimeError:
                pass

            # Generate mesh
            rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
            # rec_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

            rec_mesh.remove_duplicated_vertices()
            rec_mesh.remove_duplicated_triangles()
            rec_mesh.remove_degenerate_triangles()
            rec_mesh.remove_unreferenced_vertices()
            self._mesh = rec_mesh
            self._pcd = pcd

        self._scat_coefs = scattering or np.ones((pts.shape[0],))
        self._ref_coefs = reflectivity or np.ones((pts.shape[0],))

    def setScatteringCoeffs(self, coef):
        if coef.shape[0] != self.vertices.shape[0]:
            raise RuntimeError('Scattering coefficients must be the same size as vertex points')
        self._scat_coefs = coef

    def setReflectivityCoeffs(self, coef):
        if coef.shape[0] != self.vertices.shape[0]:
            raise RuntimeError('Reflectivity coefficients must be the same size as vertex points')
        self._ref_coefs = coef

    def visualize(self):
        o3d.visualization.draw_geometries([self._pcd, self._mesh])

    @property
    def vertices(self):
        return np.asarray(self._pcd.points)

    @property
    def triangles(self):
        return np.asarray(self._mesh.triangles)

    @property
    def normals(self):
        return np.asarray(self._mesh.vertex_normals)


class MapEnvironment(Environment):

    def __init__(self, origin, ref_llh, extent, npts_background=500, resample=False):
        lats = np.linspace(origin[0] - extent[0] / 2 / 111111, origin[0] + extent[0] / 2 / 111111, npts_background)
        lons = np.linspace(origin[1] - extent[1] / 2 / 111111, origin[1] + extent[1] / 2 / 111111, npts_background)
        lt, ln = np.meshgrid(lats, lons)
        ltp = lt.flatten()
        lnp = ln.flatten()
        e, n, u = llh2enu(ltp, lnp, getElevationMap(ltp, lnp), ref_llh)
        if resample:
            nlat, nlon, nh = resampleGrid(u.reshape(lt.shape), lats, lons, int(len(u) * .8))
            e, n, u = llh2enu(nlat, nlon, nh + ref_llh[2], ref_llh)
        super().__init__(np.array([e, n, u]).T)