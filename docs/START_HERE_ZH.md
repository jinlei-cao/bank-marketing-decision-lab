# Jinlei Cao · GitHub 项目发布与运行指引

**目标仓库名称：** `bank-marketing-decision-lab`  
**个人主页：** https://github.com/jinlei-cao  
**项目用途：** 商业分析、市场分析和银行营销决策的量化研究作品集。

## 1. 首次使用：无需真实数据也能运行

电脑安装 Python 3.10 或以上版本，进入解压后的项目文件夹，在终端运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[app,dev]'
banklab demo
streamlit run app/streamlit_app.py
```

Windows PowerShell 的环境激活命令为 `.venv\Scripts\Activate.ps1`。演示数据完全由程序随机生成，用于检查代码与界面；不代表真实银行市场结论。

## 2. 运行正式 UCI 数据研究

```bash
banklab download
banklab train --data data/raw/bank-additional-full.csv --output reports/latest
streamlit run app/streamlit_app.py
```

在网页左侧把路径改成 `reports/latest`。首次数据下载需要可访问 UCI 的网络环境。`reports/latest` 将生成模型比较、指标、图表及结果说明。实际研究结论只能根据本地运行产生的结果填写。

## 3. 上传 GitHub

1. 登录 Chloe 的 GitHub 账户 `jinlei-cao`。
2. 选择 **New repository**，名称填写 `bank-marketing-decision-lab`，设置为 **Public**。
3. 新建仓库时不要额外勾选初始化 README / License，避免首次推送冲突。
4. 推荐在项目根目录运行：

```bash
git init
git add .
git commit -m "Initial reproducible bank marketing research toolkit"
git branch -M main
git remote add origin https://github.com/jinlei-cao/bank-marketing-decision-lab.git
git push -u origin main
```

也可以通过 GitHub 页面 **Add file → Upload files** 上传项目文件。`.github/workflows/tests.yml` 以及隐藏文件需要保持原路径，使用 Git 命令通常更可靠。

## 4. 正式公开前核对

- [ ] 确认全部脚本能运行，并能解释数据泄漏、时间切分和 PR-AUC。
- [ ] 使用真实 UCI 数据运行正式实验，留存命令、环境和结果截图。
- [ ] 在 README 添加真实运行的指标（明确数据日期和限制）。
- [ ] 检查代码、项目论文和可视化结果一致。
- [ ] 明确 synthetic demo 与 real research 的区别。
- [ ] 再决定是否在硕士申请 CV 中引用该 GitHub 地址。

## 5. 学术诚信和隐私

该初始代码包由 AI 协助构建，建议 Chloe 亲自运行、审阅、理解与持续修改。课程申请中应如实描述个人实际贡献。公开仓库不含个人手机号、私人电子邮件、真实银行客户记录或无法核实的研究成绩。
