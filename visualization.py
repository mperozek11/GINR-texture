##what we want for visualization:
#textured ball-image saving?
#smooth type-variable
#smoothed ball-image saving?
#dataset used-variable
#norms/vectors-variable
#loss-list of best_loss
#inference
#output-image saving?
#texture-mesh
#smoothed-mesh
#smooth_type-string
#dataset_name-string
#dataset_type-string
#losses-list
#output-mesh
def visualization(texture, smoothed, smooth_type,dataset_name,dataset_type,losses,output):
    ###'''textured_scene=texture.scene()
   ### textured_image = textured_scene.save_image(resolution=(1080,1080))
    ###smoothed_scene=texture.scene()
    ###smoothed_image = smoothed_scene.save_image(resolution=(1080,1080))
   ### output_scene=output.scene()
   ### output_image = output_scene.save_image(resolution=(1080,1080))'''


    vis={'textured': texture,
        'smoothed': smoothed,
        'smooth_type': smooth_type,
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'losses':losses,
        'output_image':output
    }
    return vis

