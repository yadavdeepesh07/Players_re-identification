1. Approach and Methodology
Our solution performs player tracking from a sports video using a multi-stage pipeline that integrates:

YOLOv11 (Ultralytics) for detecting players frame-by-frame.

Kalman Filter for predicting the position of detected players even when occluded.

Re-Identification (Re-ID) via a lightweight ResNet18-based feature extractor to recover players after re-entry or long occlusion.

A custom TrackManager that combines detections with predictions, handles occlusions using a disappeared count, and maintains persistent IDs.

The full pipeline includes:

Preprocessing video input.

Detecting players per frame using YOLO.

Updating track logic and predicting positions for missing players.

Matching appearance features for re-entry using cosine similarity.

Logging all ID transitions and matches in logs/reid_log.txt.

Saving the annotated video with player boxes and IDs.

2. Techniques Tried and Outcomes

Technique	                                                                    Outcome
YOLOv11 for detection	                                           Accurate player bounding boxes with real-time inference.

Kalman Filter tracking	                                           Smooth short-term prediction even when players go missing briefly.

Re-ID via feature vectors	                                       Effective in re-assigning same IDs after re-entry with high similarity.

Logging to file                                                	   All transitions and track changes are traceable via text logs.

Output video storage	                                           Processed video is saved for later evaluation and presentation.



3. Challenges Encountered
Model Loading Issues: Missing pretrained weights required offline downloads due to SSL failures.

Re-ID feature loading: Torchvision threw warnings regarding deprecated pretrained=True—resolved by switching to weights="DEFAULT".

Kalman state tuning: The filter required trial-and-error to balance motion prediction smoothness vs. adaptability.

Occlusion timing: Finding the right threshold to declare an ID as "gone" was non-trivial and dataset-dependent.

4. Future Work / Incompleteness
While the core pipeline is functional, further work could include:

Better Re-ID Embeddings: Use a pretrained Re-ID model (e.g., OSNet or Market1501-trained net) for better feature accuracy.

Multi-camera Support: Extend to multi-view or multi-camera setups.

Real-time Optimization: Reduce lag by optimizing detection + re-ID using GPU batching.

Metrics Dashboard: Add a dashboard (e.g., with matplotlib or wandb) for real-time tracking analysis.