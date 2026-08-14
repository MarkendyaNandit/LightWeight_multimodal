# Master Prompt for Project Hand-off

Copy the text below and paste it directly to the AI agent on your new laptop. This prompt will instruct the AI to systematically fix the pipeline and train the models on all datasets.

---

**Master Instruction for Multimodal Anomaly Diagnostics System Integration:**

Hello! We are working on a Multimodal Industrial Anomaly Diagnostics project. We have a pipeline that combines an RGB Encoder, Depth Encoder, GACM Fusion model, and CLIP/OCTA text alignment. 

Currently, the model works perfectly for the 10 MVTec 3D categories, but **it is failing on our 3 custom categories (`phone_screen`, `car_metal`, `pcb`) because the visual encoders and fusion models were never actually trained on them.**

Please execute the following step-by-step master plan to fix the pipeline, train the missing datasets, and get the web server running flawlessly for all 13 categories. Do this using your tools and write scripts as needed.

### Phase 1: Update Dataloaders for Custom Datasets
The training scripts (RGB, Depth, Fusion) currently hardcode the MVTec 3D dataset loader. You need to write a `UnifiedAnomalyDataset` class that can seamlessly load both MVTec 3D categories AND our custom datasets.
Here is the mapping for the custom datasets relative to the project root:
1. **Phone Screen (`phone_screen`)**: 
   - Normal: `archive/good/*.png`
   - Defect: `archive/scratch/*.jpg`
2. **PCB (`pcb`)**:
   - Normal: `DeepPCB-master/DeepPCB-master/PCBData/**/*_temp.jpg`
   - Defect: `DeepPCB-master/DeepPCB-master/PCBData/**/*_test.jpg`
3. **Car Metal (`car_metal`)**:
   - Located in `archive (1)/NEU-DET/train/images`. Use `crazing` as the "Normal" class, and `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches` as the "Defect" classes.
   *Note: Since these are 2D datasets, for Depth, simply convert the RGB image to Grayscale and pass it as a 3-channel depth map.*

### Phase 2: Fine-tune the Encoders & Fusion Model
Once the unified dataloader is built, you need to fine-tune the models so they learn the feature manifolds of the 3 custom categories.
1. Fine-tune the **RGB Encoder** (`sdc_project/feature_head.py`) on all 13 categories using a classification/contrastive loss.
2. Fine-tune the **Depth Encoder** (`depth_encoder_share/depth_encoder/models/depth_encoder.py`) similarly.
3. Fine-tune the **GACM Visual Fusion Model** (`multimodal_fusion_pipeline/train_fusion.py`) using features from all 13 categories. Save the updated checkpoint as `best_fusion_model_finetuned.pth`.

### Phase 3: Recompute Manifold Means & PatchCore Banks
The system relies on distance to the "Normal" feature mean, but the means for the custom classes were corrupted/missing.
1. Write a script to loop through the **Normal** images of **ALL 13 categories**, run them through the fine-tuned RGB + Depth + Fusion pipeline, and calculate the `L2-normalized Mean Feature Vector` for each category.
2. Save these to `multimodal_fusion_pipeline/outputs/checkpoints/all_category_means.json`.
3. (Optional but recommended): Rebuild the PatchCore coreset memory banks (`patchcore_coreset_banks.pt`) to include memory banks for the 3 custom categories.

### Phase 4: Calibrate Thresholds & Update the Web App
1. Write an evaluation script that runs inference on a validation set of Normal and Defect images for ALL 13 categories.
2. Compute the optimal `manifold_dist` threshold for each category using Youden’s J-Index to maximize accuracy.
3. Update `web_app/main.py` with these new calibrated thresholds in the `threshold_map`.
4. Ensure `web_app/main.py` correctly loads the new unified models and `all_category_means.json`.
5. Start the web server (`python web_app/main.py`), resolve any port conflicts, and verify that the API correctly predicts anomalies for the 3 custom categories with high accuracy!

*Please follow these instructions sequentially. Feel free to verify the files and directories first, then create an implementation plan before executing.*


