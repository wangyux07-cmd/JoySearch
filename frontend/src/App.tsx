import { useEffect, useState } from "react";
import type { FormEvent } from "react";

type Result = {
  catalog_id: string;
  item_id: string;
  title: string | null;
  rank: number;
  score: number;
  sources: string[];
  metadata: { product_class?: string };
};
type Ranking = {
  algorithm: string;
  model_version: string;
  results: Result[];
  latency_ms: number;
  fallback_reason?: string | null;
};
type User = { user_id: string; history_count: number };
type Metrics = {
  hr_at_10: number;
  ndcg_at_10: number;
  catalog_coverage_at_10: number;
  fallback_rate: number;
};
type Report = {
  status: string;
  reason?: string;
  run_id?: string;
  search?: {
    sample_count: number;
    catalog_size: number;
    metrics: Record<string, number>;
    algorithms?: Record<string, { metrics: Record<string, number> }>;
  };
  recommendation?: {
    users_evaluated: number;
    catalog_size: number;
    unavailable_test_item_rate: number;
    algorithms: Record<string, Metrics>;
  };
};

async function api<T>(
  path: string,
  body?: object,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(path, {
    signal,
    ...(body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {}),
  });
  if (!response.ok) {
    const error = response.headers
      .get("content-type")
      ?.includes("application/json")
      ? await response.json()
      : {
          detail: `HTTP ${response.status}：无法访问本地 API，请确认后端已启动。`,
        };
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : JSON.stringify(error.detail),
    );
  }
  return response.json();
}

const labels: Record<string, string> = {
  ndcg_at_10: "NDCG@10",
  mrr_at_10: "MRR@10",
  judged_recall_at_100: "JudgedRecall@100",
  judgment_coverage_at_10: "JudgmentCoverage@10",
};

export default function App() {
  const [tab, setTab] = useState<"search" | "rec" | "metrics">("search");
  const [query, setQuery] = useState("salon chair");
  const [users, setUsers] = useState<User[]>([]);
  const [user, setUser] = useState("");
  const [algorithm, setAlgorithm] = useState("itemcf");
  const [searchAlgorithm, setSearchAlgorithm] = useState("bm25");
  const [searchAlgorithms, setSearchAlgorithms] = useState<string[]>([]);
  const [recAlgorithms, setRecAlgorithms] = useState<string[]>([]);
  const [capabilityError, setCapabilityError] = useState("");
  const [ranking, setRanking] = useState<Ranking | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [userError, setUserError] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [reportError, setReportError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    api<{ users: User[] }>("/api/v1/users/demo", undefined, controller.signal)
      .then((data) => {
        setUsers(data.users);
        setUser(data.users[0]?.user_id ?? "");
      })
      .catch((cause: Error) => {
        if (cause.name !== "AbortError") setUserError(cause.message);
      });
    api<Report>("/api/v1/reports/summary", undefined, controller.signal)
      .then(setReport)
      .catch((cause: Error) => {
        if (cause.name !== "AbortError") setReportError(cause.message);
      });
    fetch("/readyz", { signal: controller.signal })
      .then(async (response) => {
        if (![200, 503].includes(response.status))
          throw new Error(`HTTP ${response.status}`);
        const data: {
          search_algorithms?: unknown;
          recommendation_algorithms?: unknown;
        } = await response.json();
        const algorithms = data.search_algorithms;
        if (
          !Array.isArray(algorithms) ||
          !algorithms.every((a) =>
            ["bm25", "dense", "hybrid", "ltr", "query_hybrid"].includes(a),
          )
        ) {
          throw new Error("后端搜索能力合同不匹配，请重启当前版本 API。");
        }
        setSearchAlgorithms(algorithms);
        const recommendations = data.recommendation_algorithms;
        if (
          !Array.isArray(recommendations) ||
          !recommendations.every((a) =>
            ["popularity", "itemcf", "two_tower"].includes(a),
          )
        ) {
          throw new Error("后端推荐能力合同不匹配，请重启当前版本 API。");
        }
        setRecAlgorithms(recommendations);
      })
      .catch((cause: Error) => {
        if (cause.name !== "AbortError") setCapabilityError(cause.message);
      });
    return () => controller.abort();
  }, []);

  function changeTab(next: typeof tab) {
    setTab(next);
    setRanking(null);
    setError("");
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setRanking(null);
    try {
      const result =
        tab === "search"
          ? await api<Ranking>("/api/v1/search", {
              catalog_id: "wands_search",
              query,
              top_k: 10,
              algorithm: searchAlgorithm,
            })
          : await api<Ranking>("/api/v1/recommend", {
              catalog_id: "amazon_baby_rec",
              user_id: user || null,
              top_k: 10,
              algorithm,
              exclude_seen: true,
            });
      setRanking(result);
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "请求失败，请检查 API 服务。",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header className="masthead">
        <a className="brand" href="/" aria-label="JoySearch 首页">
          <span className="brandmark" aria-hidden="true">
            J
          </span>
          JoySearch
        </a>
        <span className="project-note">搜索与推荐实验室</span>
        <span className="local-tag">本地 CPU 运行</span>
      </header>
      <nav aria-label="功能导航">
        {(
          [
            ["search", "Search · 搜索"],
            ["rec", "For You · 推荐"],
            ["metrics", "Evaluation · 评测"],
          ] as const
        ).map(([value, name]) => (
          <button
            key={value}
            type="button"
            disabled={busy}
            aria-current={tab === value ? "page" : undefined}
            onClick={() => changeTab(value)}
          >
            {name}
          </button>
        ))}
      </nav>
      <main>
        {tab !== "metrics" ? (
          <>
            <div className="page-heading">
              <div>
                <h1>
                  {tab === "search"
                    ? "找到相关的商品。"
                    : "从真实历史中发现偏好。"}
                </h1>
                <p>
                  {tab === "search"
                    ? "在 WANDS 家居商品库中检索，查看真实标题与排序。"
                    : "选择匿名实验用户，对照个性化推荐与热门基线。"}
                </p>
              </div>
              <span className="catalog">
                {tab === "search"
                  ? "WANDS Search Catalog"
                  : "Amazon Recommendation Catalog"}
              </span>
            </div>
            <div className="workspace">
              <aside>
                <form onSubmit={submit}>
                  {tab === "search" ? (
                    <>
                      <label htmlFor="query">搜索词</label>
                      <input
                        disabled={busy}
                        id="query"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        maxLength={500}
                        required
                      />
                      <label htmlFor="search-algorithm">检索算法</label>
                      <select
                        id="search-algorithm"
                        disabled={busy}
                        value={searchAlgorithm}
                        onChange={(e) => setSearchAlgorithm(e.target.value)}
                      >
                        {[
                          ["bm25", "BM25 · 关键词"],
                          ["dense", "Dense · 语义检索"],
                          ["hybrid", "Hybrid · RRF 融合"],
                          ["ltr", "LambdaMART · 学习排序"],
                            ["query_hybrid", "Query Hybrid · 查询理解"],
                        ].map(([value, label]) => (
                          <option
                            key={value}
                            value={value}
                            disabled={!searchAlgorithms.includes(value)}
                          >
                            {label}
                            {searchAlgorithms.includes(value)
                              ? ""
                              : " · 未就绪"}
                          </option>
                        ))}
                      </select>
                      {capabilityError && (
                        <p className="error" role="alert">
                          {capabilityError}
                        </p>
                      )}
                      <p className="helper">
                        英文查询与原始商品文本匹配。可尝试 wooden desk、sofa 或
                        salon chair。
                      </p>
                    </>
                  ) : (
                    <>
                      <label htmlFor="user">匿名用户</label>
                      <select
                        disabled={busy}
                        id="user"
                        value={user}
                        onChange={(e) => setUser(e.target.value)}
                      >
                        <option value="">未知用户 · 冷启动</option>
                        {users.map((u) => (
                          <option key={u.user_id} value={u.user_id}>
                            {u.user_id} · {u.history_count} 条历史
                          </option>
                        ))}
                      </select>
                      {userError && (
                        <p role="alert" className="error">
                          {userError}
                        </p>
                      )}
                      <label htmlFor="algorithm">推荐算法</label>
                      <select
                        disabled={busy}
                        id="algorithm"
                        value={algorithm}
                        onChange={(e) => setAlgorithm(e.target.value)}
                      >
                        {[
                          ["itemcf", "ItemCF · 历史相似商品"],
                          ["popularity", "Popularity · 训练集热门"],
                          ["two_tower", "Two-Tower · 历史向量召回"],
                        ].map(([value, label]) => (
                          <option
                            key={value}
                            value={value}
                            disabled={!recAlgorithms.includes(value)}
                          >
                            {label}
                            {recAlgorithms.includes(value) ? "" : " · 未就绪"}
                          </option>
                        ))}
                      </select>
                      {capabilityError && (
                        <p className="error" role="alert">
                          {capabilityError}
                        </p>
                      )}
                      <p className="helper">
                        排除训练历史中已评分为正向的商品。未知用户使用热门推荐。
                      </p>
                    </>
                  )}
                  <button
                    className="primary"
                    disabled={
                      busy ||
                      (tab === "rec" && !recAlgorithms.includes(algorithm)) ||
                      (tab === "search" &&
                        (!query.trim() ||
                          !searchAlgorithms.includes(searchAlgorithm)))
                    }
                    type="submit"
                  >
                    {busy
                      ? "正在计算…"
                      : tab === "search"
                        ? "搜索商品"
                        : "生成推荐"}
                  </button>
                </form>
                <div className="data-note">
                  <h2>关于这份数据</h2>
                  <p>
                    {tab === "search"
                      ? "真实商品和稀疏相关性标注。没有价格、库存或卖家信息。"
                      : "正向评分是偏好代理，不代表点击或购买。仅显示真实 Amazon 商品 ID，metadata unavailable。"}
                  </p>
                  <p>两个商品库相互独立，不共享商品或用户身份。</p>
                </div>
              </aside>
              <section
                className="results"
                aria-label="排序结果"
                aria-busy={busy}
              >
                <div className="results-header">
                  <h2>排序结果</h2>
                  {ranking && (
                    <span>
                      {ranking.results.length} 条 ·{" "}
                      {ranking.latency_ms.toFixed(1)} ms · {ranking.algorithm}
                    </span>
                  )}
                </div>
                <div aria-live="polite">
                  {busy && <p className="empty">正在从本地模型读取结果…</p>}
                  {error && (
                    <p role="alert" className="error">
                      {error}
                    </p>
                  )}
                  {!busy && !error && !ranking && (
                    <div className="empty">
                      <span className="empty-symbol" aria-hidden="true">
                        ⌕
                      </span>
                      <h3>
                        {tab === "search"
                          ? "从一个查询开始"
                          : "看看不同历史带来的差异"}
                      </h3>
                      <p>
                        {tab === "search"
                          ? "输入关键词后点击“搜索商品”。"
                          : "选择用户和算法，然后点击“生成推荐”。"}
                      </p>
                    </div>
                  )}
                  {ranking?.fallback_reason && (
                    <p className="notice">
                      {ranking.fallback_reason === "cold_user"
                        ? "冷用户：使用训练集热门商品。"
                        : "相似邻居不足：部分位置由训练集热门商品补全。"}{" "}
                      <code>{ranking.fallback_reason}</code>
                    </p>
                  )}
                  {ranking && !ranking.results.length && (
                    <p className="empty">
                      没有匹配结果，请尝试其他英文关键词。
                    </p>
                  )}
                </div>
                {ranking && (
                  <ol className="result-list">
                    {ranking.results.map((item) => (
                      <li key={item.item_id}>
                        <span className="rank">
                          {item.rank.toString().padStart(2, "0")}
                        </span>
                        <div className="item-copy">
                          <h3>{item.title ?? `Product ${item.item_id}`}</h3>
                          <p>
                            {item.title
                              ? item.metadata.product_class || "商品类别未提供"
                              : "metadata unavailable"}
                          </p>
                          <div className="item-detail">
                            <code>{item.item_id}</code>
                            <span>{item.sources.join(", ")}</span>
                            <span>score {item.score.toFixed(3)}</span>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ol>
                )}
              </section>
            </div>
          </>
        ) : (
          <section className="evaluation">
            <div className="page-heading">
              <div>
                <h1>让结果可以被核验。</h1>
                <p>
                  固定测试集、独立商品库、完整训练候选集合。数值来自最近一次实际运行。
                </p>
              </div>
            </div>
            {reportError && (
              <p role="alert" className="error">
                {reportError}
              </p>
            )}
            {!report && !reportError && <p role="status">正在读取评测报告…</p>}
            {report?.status !== "completed" && report && (
              <p className="notice">尚未评测：{report.reason}</p>
            )}
            {report?.search && (
              <>
                <h2>WANDS · 搜索算法对照</h2>
                <p>
                  {report.search.sample_count} 个测试查询 /{" "}
                  {report.search.catalog_size.toLocaleString()} 件商品
                </p>
                <div className="table-wrap">
                  <table>
                    <caption>搜索相关性：未标注结果按零相关性计分</caption>
                    <thead>
                      <tr>
                        <th>算法</th>
                        {Object.keys(report.search.metrics).map((key) => (
                          <th key={key}>{labels[key] ?? key}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(
                        report.search.algorithms ?? {
                          bm25: { metrics: report.search.metrics },
                        },
                      ).map(([algorithm, values]) => (
                        <tr key={algorithm}>
                          <th scope="row">{algorithm}</th>
                          {Object.keys(report.search!.metrics).map((key) => (
                            <td key={key}>{values.metrics[key].toFixed(4)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
            {report?.recommendation && (
              <>
                <h2>Amazon · 推荐算法对照</h2>
                <p>
                  {report.recommendation.users_evaluated.toLocaleString()}{" "}
                  个相同测试用户 /{" "}
                  {report.recommendation.catalog_size.toLocaleString()}{" "}
                  件训练商品
                </p>
                <div className="table-wrap">
                  <table>
                    <caption>
                      离线正向评分命中与覆盖率；不等同于业务效果
                    </caption>
                    <thead>
                      <tr>
                        <th>算法</th>
                        <th>HR@10</th>
                        <th>NDCG@10</th>
                        <th>CatalogCoverage@10</th>
                        <th>FallbackRate</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(report.recommendation.algorithms).map(
                        ([key, value]) => (
                          <tr key={key}>
                            <th scope="row">{key}</th>
                            <td>{value.hr_at_10.toFixed(4)}</td>
                            <td>{value.ndcg_at_10.toFixed(4)}</td>
                            <td>{value.catalog_coverage_at_10.toFixed(4)}</td>
                            <td>{value.fallback_rate.toFixed(4)}</td>
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
                <p className="notice">
                  训练库外的测试目标占{" "}
                  {(
                    report.recommendation.unavailable_test_item_rate * 100
                  ).toFixed(2)}
                  %，全部保留为
                  misses。覆盖率高不代表命中率高；请对照各算法的实测结果。
                </p>
              </>
            )}
            {report?.run_id && (
              <p className="run-id">
                Run ID: <code>{report.run_id}</code>
              </p>
            )}
          </section>
        )}
      </main>
      <footer>JoySearch · 搜索与推荐实验室</footer>
    </div>
  );
}
