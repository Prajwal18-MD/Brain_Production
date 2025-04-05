from django.shortcuts import render
import os
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import SimpleITK as sitk
from nilearn.masking import compute_brain_mask
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
import tensorflow as tf  # for loading your model
import joblib  # for loading your scaler

def home(request):
    return render(request, 'home.html')

def upload_mri(request):
    success = False
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)  # Saved in 'media/'
        success = True

    # Get the list of uploaded files to show on the frontend (for selection)
    uploaded_files = os.listdir(os.path.join(settings.MEDIA_ROOT))
    nii_files = [f for f in uploaded_files if f.endswith('.nii') or f.endswith('.nii.gz')]

    return render(request, 'upload.html', {'success': success, 'nii_files': nii_files})

def predict(request):
    plot_div = None
    selected_file = None
    nii_files = os.listdir(os.path.join(settings.MEDIA_ROOT))
    nii_files = [f for f in nii_files if f.endswith('.nii') or f.endswith('.nii.gz')]

    if request.method == 'POST':
        selected_file = request.POST.get('selected_file')
        if selected_file:
            nii_file_path = os.path.join(settings.MEDIA_ROOT, selected_file)

            # Load MRI scan
            img = nib.load(nii_file_path)
            data = img.get_fdata()

            # Normalize intensity values
            data = (data - np.min(data)) / (np.max(data) - np.min(data))

            # Get valid slice indices
            non_zero_x = np.where(data.sum(axis=(1, 2)) > 0)[0]
            non_zero_y = np.where(data.sum(axis=(0, 2)) > 0)[0]
            non_zero_z = np.where(data.sum(axis=(0, 1)) > 0)[0]

            if len(non_zero_x) == 0 or len(non_zero_y) == 0 or len(non_zero_z) == 0:
                return render(request, 'prediction.html', {'error': "Invalid or empty MRI data.", 'nii_files': nii_files})

            # Define slices (10 slices for each plane)
            axial_slices = np.linspace(non_zero_x[0], non_zero_x[-1], 10, dtype=int)  
            coronal_slices = np.linspace(non_zero_y[0], non_zero_y[-1], 10, dtype=int)
            sagittal_slices = np.linspace(non_zero_z[0], non_zero_z[-1], 10, dtype=int)
            slice_count = len(axial_slices)

            # Create 3D figure
            fig = go.Figure()
            slices = []

            # Axial slices
            for s in axial_slices:
                slices.append(go.Surface(
                    z=np.ones_like(data[s, :, :]) * s,
                    x=np.linspace(0, data.shape[2], data.shape[2]),
                    y=np.linspace(0, data.shape[1], data.shape[1]),
                    surfacecolor=data[s, :, :],
                    colorscale="gray",
                    showscale=False,
                    opacity=0.7,
                    visible=False
                ))

            # Coronal slices
            for s in coronal_slices:
                slices.append(go.Surface(
                    z=np.linspace(0, data.shape[0], data.shape[0]),
                    x=np.ones_like(data[:, s, :]) * s,
                    y=np.linspace(0, data.shape[2], data.shape[2]),
                    surfacecolor=data[:, s, :].T,
                    colorscale="gray",
                    showscale=False,
                    opacity=0.7,
                    visible=False
                ))

            # Sagittal slices
            for s in sagittal_slices:
                slices.append(go.Surface(
                    z=np.linspace(0, data.shape[0], data.shape[0]),
                    x=np.linspace(0, data.shape[1], data.shape[1]),
                    y=np.ones_like(data[:, :, s]) * s,
                    surfacecolor=data[:, :, s].T,
                    colorscale="gray",
                    showscale=False,
                    opacity=0.7,
                    visible=False
                ))

            fig.add_traces(slices)

            # Dropdown buttons
            buttons = [
                dict(
                    label="Show All Slices",
                    method="update",
                    args=[{"visible": [True] * len(slices)}]
                ),
                dict(
                    label="Show All Axial Slices",
                    method="update",
                    args=[{"visible": [True] * slice_count + [False] * (2 * slice_count)}]
                ),
                dict(
                    label="Show All Coronal Slices",
                    method="update",
                    args=[{"visible": [False] * slice_count + [True] * slice_count + [False] * slice_count}]
                ),
                dict(
                    label="Show All Sagittal Slices",
                    method="update",
                    args=[{"visible": [False] * (2 * slice_count) + [True] * slice_count}]
                )
            ]

            # Individual slice buttons for Axial
            for i in range(slice_count):
                visibility = [False] * len(slices)
                visibility[i] = True
                buttons.append(dict(
                    label=f"Axial Slice {i+1}",
                    method="update",
                    args=[{"visible": visibility}]
                ))
            # Individual slice buttons for Coronal
            for i in range(slice_count):
                visibility = [False] * len(slices)
                visibility[slice_count + i] = True
                buttons.append(dict(
                    label=f"Coronal Slice {i+1}",
                    method="update",
                    args=[{"visible": visibility}]
                ))
            # Individual slice buttons for Sagittal
            for i in range(slice_count):
                visibility = [False] * len(slices)
                visibility[2 * slice_count + i] = True
                buttons.append(dict(
                    label=f"Sagittal Slice {i+1}",
                    method="update",
                    args=[{"visible": visibility}]
                ))

            fig.update_layout(
                title="Interactive 3D MRI Visualization",
                updatemenus=[dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                )],
                scene=dict(
                    xaxis=dict(title="X-Axis", visible=True),
                    yaxis=dict(title="Y-Axis", visible=True),
                    zaxis=dict(title="Z-Axis", visible=True),
                ),
                margin=dict(l=0, r=0, t=40, b=0),
            )

            plot_div = plot(fig, output_type='div', include_plotlyjs=True)

    return render(request, 'prediction.html', {
        'plot_div': plot_div,
        'nii_files': nii_files,
        'selected_file': selected_file
    })

# ------------------ New Code for Results View ------------------

def bias_field_correction_in_memory(nifti_file, do_bias_correction=True):
    """
    Applies N4 Bias Field Correction in memory to reduce intensity inhomogeneity.
    Returns a nibabel image of the (corrected) data.
    """
    sitk_image = sitk.ReadImage(nifti_file)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
    
    if do_bias_correction:
        mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50])
        corrected_sitk = corrector.Execute(sitk_image, mask_image)
    else:
        corrected_sitk = sitk_image

    corrected_array = sitk.GetArrayFromImage(corrected_sitk)
    orig_img = nib.load(nifti_file)
    corrected_img = nib.Nifti1Image(corrected_array, affine=orig_img.affine, header=orig_img.header)
    return corrected_img

def segment_tissues_kmeans(img, n_clusters=3, n_init=10):
    """
    Segments the brain into CSF, GM, and WM using k-means clustering.
    Returns a 3D numpy array with segmentation labels and the cluster-to-tissue mapping.
    """
    data = img.get_fdata()
    brain_mask_img = compute_brain_mask(img)
    mask = brain_mask_img.get_fdata().astype(bool)
    brain_voxels = data[mask]
    brain_voxels_smoothed = gaussian_filter(brain_voxels, sigma=1)
    X = brain_voxels_smoothed.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)
    tissue_mapping = {
        sorted_indices[0]: 'CSF',
        sorted_indices[1]: 'GM',
        sorted_indices[2]: 'WM'
    }
    mapping_numeric = {'CSF': 1, 'GM': 2, 'WM': 3}
    seg_labels = np.array([mapping_numeric[tissue_mapping[label]] for label in labels])
    seg_data = np.zeros(data.shape, dtype=np.int32)
    seg_data[mask] = seg_labels
    return seg_data, tissue_mapping

def compute_volumes(seg_data, img):
    """
    Computes tissue volumes (in cm³) for CSF, GM, WM, and the Total Intracranial Volume (TIV).
    """
    voxel_sizes = img.header.get_zooms()[:3]
    voxel_volume_mm3 = np.prod(voxel_sizes)
    mm3_to_cm3 = 0.001
    csf_vol = np.sum(seg_data == 1) * voxel_volume_mm3 * mm3_to_cm3
    gm_vol  = np.sum(seg_data == 2) * voxel_volume_mm3 * mm3_to_cm3
    wm_vol  = np.sum(seg_data == 3) * voxel_volume_mm3 * mm3_to_cm3
    tiv = csf_vol + gm_vol + wm_vol
    return {
        'TIV_cm3': tiv,
        'CSF_Volume_cm3': csf_vol,
        'GM_Volume_cm3': gm_vol,
        'WM_Volume_cm3': wm_vol
    }

def run_prediction(nifti_path, do_bias_correction=True):
    """
    Runs the full prediction pipeline on the provided NIfTI file.
    Returns computed volumes (including predicted brain age) or an error message.
    """
    try:
        # Preprocessing: Bias correction, segmentation, and volume computation
        corrected_img = bias_field_correction_in_memory(nifti_path, do_bias_correction=do_bias_correction)
        seg_data, tissue_mapping = segment_tissues_kmeans(corrected_img)
        volumes = compute_volumes(seg_data, corrected_img)
        
        # Prepare input for the model (order must match training: [tiv, csfv, gmv, wmv])
        input_data = np.array([
            volumes['TIV_cm3'], 
            volumes['CSF_Volume_cm3'], 
            volumes['GM_Volume_cm3'], 
            volumes['WM_Volume_cm3']
        ]).reshape(1, -1)
        
        # Load the scaler and apply transformation to match training data
        scaler_path = os.path.join(settings.BASE_DIR, 'scaler.save')
        scaler = joblib.load(scaler_path)
        input_data_scaled = scaler.transform(input_data)
        
        # Load your trained model (model.keras) from the project's base directory
        model_path = os.path.join(settings.BASE_DIR, 'model.keras')
        model = tf.keras.models.load_model(model_path)
        
        # Predict brain age (years) using the scaled input
        predicted_years = model.predict(input_data_scaled)[0][0]
        volumes['predicted_years'] = predicted_years
        
        return volumes, None
    except Exception as e:
        return None, str(e)

def results(request):
    """
    Handles the results page: lists available NIfTI files, runs prediction on POST,
    and renders the results.html template with computed brain volumes and predicted brain age.
    """
    # List uploaded MRI files
    uploaded_files = os.listdir(os.path.join(settings.MEDIA_ROOT))
    nii_files = [f for f in uploaded_files if f.endswith('.nii') or f.endswith('.nii.gz')]

    context = {
        'nii_files': nii_files,
        'selected_file': None,
        'volumes': None,
        'error': None,
    }

    if request.method == 'POST':
        selected_file = request.POST.get('selected_file')
        context['selected_file'] = selected_file
        if selected_file:
            nifti_path = os.path.join(settings.MEDIA_ROOT, selected_file)
            volumes, error = run_prediction(nifti_path, do_bias_correction=True)
            if volumes:
                context['volumes'] = volumes
            else:
                context['error'] = error

    return render(request, 'results.html', context)
