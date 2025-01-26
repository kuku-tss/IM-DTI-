Configuring
===========

Below is a sample configuration file for the `train.py` script.

.. code-block:: yaml
    
    # Logging and Paths
    wandb_proj: NoSigmoidTest # Weights and Biases project to log results to.
    wandb_save: True # Whether or not to log to Weights and Biases.
    log_file: ./logs/scratch_testing.log # Location of log file
    model_save_dir: ./best_models # Location to save best models
    data_cache_dir: /data/cb/samsl/Adapting_PLM_DTI/dataset # Location of downloaded data (use `conplex_dti download`)

    # Misc
    device: 0 # CUDA device to use for training
    replicate: 0 # Random seed
    verbosity: 3 # Verbosity level for logging

    # Task and Dataset
    task: davis # Benchmark task - one of "davis", "bindingdb", "biosnap", "biosnap_prot", "biosnap_mol", "dti_dg"
    contrastive_split: within # Train/test split for contrastive learning is between target classes or within target classes

    # Model and Featurizers
    drug_featurizer: MorganFeaturizer # Featurizer for small molecule drug SMILES (see `conplex_dti.featurizer` documentation)
    target_featurizer: ProtBertFeaturizer # Featurizer for protein sequences (see `conplex_dti.featurizer` documentation)
    model_architecture: SimpleCoembeddingNoSigmoid # Model architecture (see `conplex_dti.models` documentation)
    latent_dimension: 1024 # Dimension of shared co-embedding space
    latent_distance: "Cosine" # Distance metric to use in learned co-embedding space

    # Training
    epochs: 50 # Number of epochs to train for

    ## Batching
    batch_size: 32 # Size of batch for binary data set
    contrastive_batch_size: 256 # Size of batch for contrastive data set
    shuffle: True # Whether to shuffle training data before batching
    num_workers: 0 # Number of workers for PyTorch DataLoader
    every_n_val: 1 # How often to run validation during training (epochs)

    ## Learning Rate
    lr: 1e-4 # Learning rate for binary training
    lr_t0: 10 # With annealing, reset learning rate to initial value after this many epochs for binary traniing

    ## Contrastive
    contrastive: True # Whether to use contrastive learning
    clr: 1e-5 # Learning rate for contrastive training
    clr_t0: 10 # With annealing, reset learning rate to initial value after this many epochs for contrastive training

    ## Margin
    margin_fn: 'tanh_decay' # Margin annealing function to use for contrastive triplet distance loss
    margin_max: 0.25 # Maximum margin value
    margin_t0: 10 # With annealing, reset margin to initial value after this many epochs for contrastive training
