### Preprocessing file execution order
Beforehand, place unzipped datasets in base project directory and rename by adding "miche_" prefix to miche datasets, then:

1. run files:
   preprocess_all_mmu.py,  
   preprocess_all_miche.py,  
   preprocess_all_ubiris.py (for both parts of the dataset)

   this will take several hours
2. run file change_mmu_structure.py
3. run file filter_out_bad_samples.py
4. run file create_final_merged_data_folder.py
5. run file sample_classes_for_feature_extraction_testing.py
6. run file split_data_into_train_test.py (for all datasets and merged dataset separately)