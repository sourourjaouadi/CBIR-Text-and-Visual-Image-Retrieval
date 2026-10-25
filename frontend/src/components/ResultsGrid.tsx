import type { Result } from '../api/client';

type Props = { results: Result[]; onChoose: (result: Result) => void };

export default function ResultsGrid({ results, onChoose }: Props) {
  const ordered = [...results].sort((a, b) => a.distance - b.distance || a.image_id - b.image_id);
  if (!ordered.length) return <div className="empty-state">No matching images yet. Try another keyword or query image.</div>;
  return <div className="grid">{ordered.map((item, index) => (
    <button className="card" key={item.image_id} onClick={() => onChoose(item)} title="Search again using this image">
      <img src={item.url} alt={`Image ${item.image_id}`} loading="lazy"/>
      <div className="card-meta"><span className="rank-badge">#{item.rank || index + 1}</span><span className="distance-badge">d {item.distance.toFixed(4)}</span></div>
      {item.keywords?.length ? <span className="card-keywords">{item.keywords.slice(0, 3).join(' · ')}</span> : null}
    </button>
  ))}</div>;
}
