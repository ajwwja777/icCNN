# 平均池化降维，将特征向量从 (6, 512, 14, 14) 降维到 (6, 512)
all_feature_pooled = np.mean(all_feature, axis=(2, 3))

# 执行t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
all_feature_tsne = tsne.fit_transform(all_feature_pooled)

# 可视化
plt.scatter(all_feature_tsne[:, 0], all_feature_tsne[:, 1], s=1)

最后的到都聚类效果不是很好，解释一下这些参数是什么意思，并改善