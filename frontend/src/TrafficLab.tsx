import { useEffect, useState } from "react";

type PolicyMetrics = {
  policy: string;
  axis: string;
  bucket: string;
  sample_count: number;
  candidate_hit_at_100: number;
  hr_at_10: number;
  ndcg_at_10: number;
  catalog_coverage_at_10: number;
  tail_list_share_at_10: number;
};
type Simulation = {
  policy: string;
  seed_count: number;
  mean_proxy_hit_rate: number;
  std_proxy_hit_rate: number;
  delta_proxy_hit_vs_fixed: number;
  mean_proxy_regret: number;
  mean_catalog_coverage: number;
  mean_tail_list_share: number;
};
type AllocationReport = {
  status: "completed" | "not_run";
  report?: {
    run_id: string;
    sample_count: number;
    catalog_size: number;
    unavailable_target_count: number;
    selected_fixed_policy: string;
    selected_adaptive_policy: string;
  };
  slices?: PolicyMetrics[];
  simulation?: Simulation[];
};

export default function TrafficLab() {
  const [data, setData] = useState<AllocationReport | null>(null);
  const [error, setError] = useState("");
  const [axis, setAxis] = useState("overall");
  const [bucket, setBucket] = useState("all");
  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/v1/reports/allocation", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<AllocationReport>;
      })
      .then(setData)
      .catch((cause: Error) => {
        if (cause.name !== "AbortError") setError(cause.message);
      });
    return () => controller.abort();
  }, []);
  const slices = data?.slices ?? [];
  const buckets = [
    ...new Set(slices.filter((s) => s.axis === axis).map((s) => s.bucket)),
  ];
  const bucketLabels: Record<string, string> = {
    all: "全部用户",
    le_2: "训练历史 1–2 条",
    le_5: "训练历史 3–5 条",
    gt_5: "训练历史 6 条及以上",
    tail: "低频目标 · 训练次数 ≤2",
    mid: "中频目标 · 训练次数 3–9",
    head: "高频目标 · 训练次数 ≥10",
    unavailable: "训练目录外目标",
  };
  return (
    <section className="evaluation" aria-label="离线流量实验">
      <div className="page-heading">
        <div>
          <h1>看清配额的收益与代价。</h1>
          <p>
            同一用户群、训练商品库与候选预算。比较召回分配、冷启动切片和离线代理奖励。
          </p>
        </div>
      </div>
      {error && (
        <p role="alert" className="error">
          报告读取失败：{error}
        </p>
      )}
      {!data && !error && <p role="status">正在读取离线实验结果…</p>}
      {data?.status === "not_run" && (
        <p className="notice">
          尚无已完成的离线配额实验。完成评测后将在这里展示真实结果。
        </p>
      )}
      {data?.status === "completed" && data.report && (
        <>
          <p>
            {data.report.sample_count.toLocaleString()} 位测试用户 /{" "}
            {data.report.catalog_size.toLocaleString()} 件训练商品 / 最终候选
            B=100。
          </p>
          <p>
            验证集选择：固定配额{" "}
            <code>{data.report.selected_fixed_policy}</code>；请求级规则{" "}
            <code>{data.report.selected_adaptive_policy}</code>。
          </p>
          <h2>配额与冷启动对照</h2>
          <label htmlFor="slice-axis">查看维度 </label>
          <select
            id="slice-axis"
            value={axis}
            onChange={(event) => {
              const next = event.target.value;
              setAxis(next);
              setBucket(slices.find((s) => s.axis === next)?.bucket ?? "all");
            }}
          >
            <option value="overall">总体</option>
            <option value="history">训练历史长度</option>
            <option value="target">目标训练频次</option>
          </select>{" "}
          <label htmlFor="slice-bucket">用户组 </label>
          <select
            id="slice-bucket"
            value={bucket}
            onChange={(event) => setBucket(event.target.value)}
          >
            {buckets.map((b) => (
              <option key={b} value={b}>
                {bucketLabels[b] ?? b}
              </option>
            ))}
          </select>
          <div className="table-wrap">
            <table>
              <caption>
                每种策略使用同一组用户；覆盖率分母始终为完整训练目录，低频商品不等于新品。
              </caption>
              <thead>
                <tr>
                  <th scope="col">策略</th>
                  <th scope="col">人数</th>
                  <th scope="col">CandidateHit@100</th>
                  <th scope="col">HR@10</th>
                  <th scope="col">NDCG@10</th>
                  <th scope="col">目录覆盖@10</th>
                  <th scope="col">低频列表占比@10</th>
                </tr>
              </thead>
              <tbody>
                {slices
                  .filter((s) => s.axis === axis && s.bucket === bucket)
                  .map((s) => (
                    <tr key={s.policy}>
                      <th scope="row">{s.policy}</th>
                      <td>{s.sample_count.toLocaleString()}</td>
                      <td>{s.candidate_hit_at_100.toFixed(4)}</td>
                      <td>{s.hr_at_10.toFixed(4)}</td>
                      <td>{s.ndcg_at_10.toFixed(4)}</td>
                      <td>{s.catalog_coverage_at_10.toFixed(4)}</td>
                      <td>{s.tail_list_share_at_10.toFixed(4)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="notice">
            无历史用户使用训练集 Popularity；
            {data.report.unavailable_target_count.toLocaleString()}{" "}
            个训练目录外目标保留为 miss。当前模型不提供新 ID 的内容冷启动。
          </p>
          <h2>离线策略模拟 · 评分目标命中奖励</h2>
          <p>
            每个种子重新排列同一批用户。UCB
            只接收此前所选配额的反馈；固定策略使用验证集选定的配额。
          </p>
          <div className="table-wrap">
            <table>
              <caption>
                多种子均值与样本标准差；这是离线评分代理实验，不是真实曝光日志回放或线上点击效果。
              </caption>
              <thead>
                <tr>
                  <th scope="col">策略</th>
                  <th scope="col">种子数</th>
                  <th scope="col">命中均值 ± 标准差</th>
                  <th scope="col">相对固定策略绝对差</th>
                  <th scope="col">目录覆盖</th>
                  <th scope="col">低频列表占比</th>
                  <th scope="col">代理 oracle regret</th>
                </tr>
              </thead>
              <tbody>
                {data.simulation?.map((s) => (
                  <tr key={s.policy}>
                    <th scope="row">{s.policy}</th>
                    <td>{s.seed_count}</td>
                    <td>
                      {s.mean_proxy_hit_rate.toFixed(5)} ±{" "}
                      {s.std_proxy_hit_rate.toFixed(5)}
                    </td>
                    <td>{s.delta_proxy_hit_vs_fixed.toFixed(5)}</td>
                    <td>{s.mean_catalog_coverage.toFixed(4)}</td>
                    <td>{s.mean_tail_list_share.toFixed(4)}</td>
                    <td>{s.mean_proxy_regret.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            Oracle 仅作诊断：逐请求选择能命中目标的配额形成的代理上界；不会作为
            UCB 的反馈。
          </p>
          <p className="run-id">
            Run ID: <code>{data.report.run_id}</code>
          </p>
        </>
      )}
    </section>
  );
}
