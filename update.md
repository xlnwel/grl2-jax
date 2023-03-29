#### bug记录

2023.03.22
algo.ppo.elements.utils中的compute_actor_loss函数中的policy_sample_mask判断反了

2023.03.24
1. jax.dist中MultivariateNormalDiag类中增加了属性方法scale，与tfd中计算kl_divergence的函数冲突，报错"AttributeError: 'DeviceArray' object has no attribute 'forward_log_det_jacobian'"
2. happo算法elements.trainer.py中的stepwise_sequential_opt函数中，在调用compute_temmate_log_ratio函数时，传入的agent_data中存在popart_mean和popart_std属性，这两个属性值并不可索引，和happo.elements.model.py文件中action_logprob函数里添加的data.slice(slice(None), 0)矛盾冲突，在happo的trainer.py下的compute_teammate_log_ratio函数中添加了代码，排除了这两个属性。
3. 更新版本的代码为了让智能体更新的顺序可控，使用了jax.random函数，这导致stepwise_sequential_opt函数中每次得到的idx为jnp.array，导致data_slice在mini_batch_size为1时形状和之前的版本不一致，将代码改成了idx.tolist解决了这一问题。

2023.03.25
1. 莫名修复一个由eval导致的ma-mujoco训练效果差的bug.
2. 为smac添加train_entry

2023.03.26
1. gen_data_from_expert支持自动选取expert数据
2. 把eval相关的函数整理到eval_and_log

2023.03.28
1. ma_common/train.py中函数log的参数进行了修改，老版代码是5个参数，为了兼容train_smac改成了3个参数，代码有冲突，暂时更改成了*args，根据参数个数定义参数含义。(由于在log函数中自动从agent里读取env_step和train_step, 所以把这两个参数忽略了)

2023.03.29
关闭advantage normalization, 对A>0时加一个KL.