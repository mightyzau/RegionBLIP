# Visualize objaverse point clouds

import open3d as o3d
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--npz_file', type=str)
    args = parser.parse_args()

    datas = np.load(args.npz_file)
    pc = datas['arr_0'] # [8192, 3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])


    o3d.visualization.draw_geometries([pcd],
                                  #zoom=0.3412,
                                  #front=[0.4257, -0.2125, -0.8795],
                                  #lookat=[2.6172, 2.0475, 1.532],
                                  #up=[-0.0694, -0.9768, 0.2024]
                                  )

