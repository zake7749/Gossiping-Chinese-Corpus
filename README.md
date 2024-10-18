# PTT 中文語料

🎉 News: 我們採用了類似的資料採集策略訓練了一個繁體中文的大型語言模型，模型效能與使用方式可以參考 [Kyara](https://github.com/zake7749/Kyara)

---

嗨，這裡是 PTT 中文語料集，我透過[某些假設與方法](https://github.com/zake7749/PTT-Chat-Generator) 將每篇文章化簡為問答配對，其中問題來自文章的標題，而回覆是該篇文章的推文。
可惜的是目前這份資料集的噪聲還有點大，若您有更好的方法能提取出文章的問答配對，或發現這份資料集有什麼能改進的部份，還請與我聯繫，也祝各位開發順利 :>

## 資料說明

資料集一共有兩份，您可於 [PTT-Gossiping-Corpus](https://www.kaggle.com/zake7749/pttgossipingcorpus) 或是從本專案的 `data` 資料夾裡取得。

## Gossiping-QA-Dataset.txt

蒐集了 PTT 八卦版於 2015 年至 2017 年 6 月的文章，每一行都是一個問答配對，問與答之間以一個 tab (`\t`) 區隔開，比如說

```
matlab有什麼炫砲一點的圖？	一樣的圖改一改顏色，有點半透明感覺更唬爛炫
有沒有情人節吃什麼cp值最高的八卦	吃屎啊廢話 免費的一餐
姆咪一個人守得住街亭嗎?	引來一堆肥宅穢土轉生 有機會喔
有沒有被落石砸到該反省的八卦	蔡英文執政就故意誇大報導 東森不意外
情人節該帶女朋友去哪慶祝？	用了一整年 對她好一點  送專業乾洗店吧
為什麼 聖結石 會被酸而 這群人 不會？	質感 劇本 成員 都差很多好嗎 不要拿腎結石來污辱這群人
為什麼慶祝228會被罵可是慶端午不會？	因為屈原不是台灣人，是楚國人。
有沒有戰神阿瑞斯的八卦?	爵士就是阿瑞斯 男主角最後死了
理論與實務最脫節的系	哪個系不脫節...你問最不脫節的簡單多了...
為什麼PTT這麼多人看棒球	肥宅才看棒球　系壘一堆胖子
為什麼達摩祖師傳那麼好看?	達摩從頭到尾都是被動 (別人問他問題
```

目前共有 418,202 筆問答配對，但並不是所有配對都是有效的，因為有些文章並沒有推文，這類問題的回覆會被標記為`沒有資料`(共有 650 筆)，使用時還請注意。

### Gossiping-QA-Dataset-2_0.csv

擴充自 Gossiping-QA-Dataset.txt 的新版資料集，追加了部分 2018 與 2019 年的文章，一共包含了 774,114 筆問答配對。
資料格式調整為 csv，包含了兩個 columns: `question` 和 `answer`，可用 pandas 輕鬆載入：

```
import pandas as pd
qa_corpus = pd.read_csv("data/Gossiping-QA-Dataset-2_0.csv")
qa_corpus.head()

# question	answer
# 173763	今年房市是多還是空？	有台北市的房子 再來討論
# 216985	有沒有台灣高登拉姆齊-劉一帆的八卦?	餐飲界賈伯斯
# 679552	什麼時候會覺得台灣治安很差？	台灣一年輾死三千多人
# 349323	有沒有早上起來運動很舒服的八卦？	肥宅每天都在運動,手部運動
# 42228	迪士尼還有什麼片子適合拍成真人版	想說我剛剛不是進sex版啊...
```

## Citation

If you think the dataset is helpful for your research, please cite our project:

```
@misc{
    kai-chou yang_2019,
    title={PTT-Gossiping-Corpus},
    url={https://www.kaggle.com/dsv/676336},
    DOI={10.34740/DVS/676336},
    publisher={Kaggle},
    author={Kai-Chou Yang},
    year={2019}
}
```
