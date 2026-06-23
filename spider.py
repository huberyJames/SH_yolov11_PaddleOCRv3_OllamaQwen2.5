import requests
import json
import os
import csv
from datetime import datetime

# ========== 配置 ==========
OUTPUT_DIR = r"C:\Users\Administrator\Desktop\sh\output\spider"
KEYWORD = "yolo"
PER_PAGE = 30
TOTAL_PAGES = 5

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "Mozilla/5.0"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def search_github_repos(keyword, page=1, per_page=30):
    url = "https://api.github.com/search/repositories"
    params = {
        "q": keyword,
        "sort": "stars",
        "order": "desc",
        "page": page,
        "per_page": per_page
    }
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def main():
    all_items = []
    print(f"🔍 开始爬取 GitHub 上 '{KEYWORD}' 相关仓库...")
    
    for page in range(1, TOTAL_PAGES + 1):
        print(f"\n📄 正在获取第 {page} 页...")
        try:
            result = search_github_repos(KEYWORD, page, PER_PAGE)
            items = result.get("items", [])
            if not items:
                print("⚠️ 没有更多数据")
                break
            
            for repo in items:
                all_items.append({
                    "name": repo.get("full_name"),
                    "url": repo.get("html_url"),
                    "description": repo.get("description"),
                    "stars": repo.get("stargazers_count"),
                    "forks": repo.get("forks_count"),
                    "language": repo.get("language"),
                    "updated_at": repo.get("updated_at"),
                    "topics": repo.get("topics", [])
                })
            print(f"   本页 {len(items)} 条，累计 {len(all_items)} 条")
        except Exception as e:
            print(f"❌ 第 {page} 页出错: {e}")
            break
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_path = os.path.join(OUTPUT_DIR, f"github_yolo_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON: {json_path}")
    
    # CSV
    if all_items:
        csv_path = os.path.join(OUTPUT_DIR, f"github_yolo_{timestamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=all_items[0].keys())
            writer.writeheader()
            writer.writerows(all_items)
        print(f"✅ CSV: {csv_path}")
    
    # 摘要
    summary_path = os.path.join(OUTPUT_DIR, f"github_yolo_summary_{timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"GitHub YOLO 搜索结果\n时间: {datetime.now()}\n总数: {len(all_items)}\n{'='*80}\n\n")
        for i, r in enumerate(all_items, 1):
            f.write(f"{i}. {r['name']}\n   ⭐{r['stars']} | 🍴{r['forks']} | {r['language']}\n   🔗 {r['url']}\n   📝 {r['description']}\n{'-'*60}\n")
    print(f"✅ 摘要: {summary_path}")
    print(f"\n🎉 完成！共 {len(all_items)} 条")

if __name__ == "__main__":
    main()