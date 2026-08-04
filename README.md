# GG's Notes

基于 Jekyll 与 GitHub Pages 的个人博客，访问地址：<https://ggboooy.github.io>。

## 写一篇新文章

1. 在 `_posts` 目录创建 `YYYY-MM-DD-title.md`。
2. 在文件顶部填写：

   ```yaml
   ---
   layout: post
   title: "文章标题"
   description: "一句话摘要"
   ---
   ```

3. 在下方使用 Markdown 写正文，提交到 `main` 分支后会自动发布。

## 修改个人信息

- 网站名称与简介：`_config.yml`
- 关于页面：`about.md`
- 颜色与排版：`assets/css/style.css`

## 首次启用 Pages

进入仓库的 **Settings → Pages**，把 **Build and deployment → Source** 设置为 **GitHub Actions**。随后在 **Actions** 页面手动运行工作流，或再推送一次提交。
