import open3d as o3d
import pickle
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', type=str, default="pc_region_data.pkl")
    args = parser.parse_args()

    with open(args.data_file, 'rb') as f:
        datas = pickle.load(f)
    
    point_clouds = datas['point_clouds']                    # [batch, 40000, 3]

    point_colors = datas['point_colors']
    #point_centers = datas['point_centers']                 # [batch, 512, 3]
    #pc_masks = datas['pc_masks']                           # [batch, 512]
    refer_box_corners = datas['refer_box_corners']          # [batch, 8, 3]

    if True:
        center = np.mean(refer_box_corners, axis=-2, keepdims=True)
        offset = refer_box_corners - center

        offset[..., [0, 1, 2]] = offset[..., [0, 2, 1]]
        refer_box_corners = offset + center

    b = 0
    print("caption: {}".format(datas['text'][b]))

    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(point_clouds[b])
    pcd_1.colors = o3d.utility.Vector3dVector(point_colors[b] / 255.0)
    #pcd_1.paint_uniform_color([0.5, 0.5, 0.5])


    #pcd_region = o3d.geometry.PointCloud()
    #points_region = point_clouds[b][pc_masks[b] > 0.5]
    #pcd_region.points = o3d.utility.Vector3dVector(points_region)
    #pcd_region.paint_uniform_color([1.0, 0, 0])

    points = refer_box_corners[b].tolist()
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7]
    ]
    colors = [[1.0, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)


    o3d.visualization.draw_geometries([pcd_1, line_set],
                                #zoom=0.3412,
                                #front=[0.4257, -0.2125, -0.8795],
                                #lookat=[2.6172, 2.0475, 1.532],
                                #up=[-0.0694, -0.9768, 0.2024]
                                )
