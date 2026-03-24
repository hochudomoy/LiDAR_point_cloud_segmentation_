import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import imageio.v2 as imageio
from io import BytesIO
import plotly.io as pio

def visualization_2D(lidar_df,spline_coords_list=None,color=None):
    if color is not None: color=np.asarray(color, dtype=np.float32)
    else: color=lidar_df['intensity']
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(-lidar_df['x'], -lidar_df['y'], c=color, s=1)
    if spline_coords_list is not None:
        for i, (xs, ys, zs) in enumerate(spline_coords_list):
            plt.plot(xs, ys, c='red', lw=2)
    plt.grid()
    plt.xlabel('x, m', fontsize=26)
    plt.ylabel('y, m', fontsize=26)
    plt.title('LiDAR', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.show()

def visualization_3D(xyz,labels=None,curb_lines=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    if labels is not None: color=np.asarray(labels, dtype=np.float32)
    else: color=xyz[:, 2]
    data = []
    data.append(go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=color,
            colorscale='Turbo'
        )
    ))

    if curb_lines is not None:
        for i, (xs, ys, zs) in enumerate(curb_lines):
            data.append(go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='lines',
                line=dict(color='blue',width=8),
                showlegend=False
            ))
    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.show()


def gif_2D(sequence_lidar_df, sequence_preds,sequence_curb_lines=None):
    frames = []
    for i,(lidar_df,pred) in enumerate(zip(sequence_lidar_df,sequence_preds)):
        color = np.where(pred==1,'green','red')
        fig = plt.figure(figsize=(8,8))
        plt.scatter(-lidar_df['x'], -lidar_df['y'], c=color, s=1)
        if sequence_curb_lines is not None:
            curb_lines = sequence_curb_lines[i]
            for xs, ys, zs in curb_lines:
                plt.plot(-xs, -ys, c='blue', linewidth=2)
        plt.xlim(-50,50)
        plt.ylim(-50,50)
        plt.grid()
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = img[:,:,:3]
        frames.append(img)
        plt.close()
    imageio.mimsave("ground_segmentation_2D.gif",frames,fps=10,loop=0)
    print("2D GIF saved")

def gif_3D(sequence_lidar_df, sequence_preds,sequence_curb_lines=None,fps=10,max_points=40000):

    frames = []
    all_xyz = np.concatenate([
        df[['x','y','z']].to_numpy() for df in sequence_lidar_df
    ])

    x_min, x_max = all_xyz[:,0].min(), all_xyz[:,0].max()
    y_min, y_max = all_xyz[:,1].min(), all_xyz[:,1].max()
    z_min, z_max = all_xyz[:,2].min(), all_xyz[:,2].max()

    dx = x_max - x_min
    dy = y_max - y_min
    dz = z_max - z_min
    scale = max(dx, dy, dz)

    aspectratio = dict(
        x=dx/scale,
        y=dy/scale,
        z=dz/scale
    )

    camera = dict(eye=dict(x=0.3, y=0.3, z=0.3))

    for i, (lidar_df, pred) in enumerate(zip(sequence_lidar_df, sequence_preds)):

        xyz = lidar_df[['x','y','z']].to_numpy()
        color = np.where(pred == 1, 1, 0)

        if len(xyz) > max_points:
            idx = np.random.choice(len(xyz), max_points, replace=False)
            xyz = xyz[idx]
            color = color[idx]

        if len(xyz) == 0:
            continue
        data=[]
        data.append(go.Scatter3d(
            x=xyz[:,0],
            y=xyz[:,1],
            z=xyz[:,2],
            mode='markers',
            marker=dict(
                size=1,
                color=color,
                colorscale=[[0, 'red'], [1, 'green']]
            )
        ))
        if sequence_curb_lines is not None:
            curb_lines = sequence_curb_lines[i]
            for xs, ys, zs in curb_lines:
                data.append(go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode='lines',
                    line=dict(color='blue', width=8),
                    showlegend=False
                ))
        fig = go.Figure(data=data)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[x_min, x_max]),
                yaxis=dict(range=[y_min, y_max]),
                zaxis=dict(range=[z_min, z_max]),
                aspectmode='manual',
                aspectratio=aspectratio,
                camera=camera
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )

        img_bytes = pio.to_image(
            fig,
            format="png",
            engine="kaleido"
        )

        img = imageio.imread(BytesIO(img_bytes))
        frames.append(img)

        print(f"Frame {i+1}")

    imageio.mimsave("ground_segmentation_3D.gif",frames, fps=fps, loop=0)

    print(f"3D GIF saved")