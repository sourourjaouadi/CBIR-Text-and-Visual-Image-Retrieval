import type { Descriptor } from '../api/client';
export type DescriptorInfo = { name: string; metric: string; kind?: string };
type Props = { names: DescriptorInfo[]; value: Descriptor[]; onChange: (value: Descriptor[]) => void; disabled?: boolean };
export default function DescriptorPicker({ names, value, onChange, disabled = false }: Props) {
  return <div className="descriptor-list">{names.map(({ name, kind }) => {
    const selected = value.find(item => item.name === name);
    return <label key={name}><input type="checkbox" disabled={disabled} checked={!!selected} onChange={e => onChange(e.target.checked ? [...value, { name, weight: 1 }] : value.filter(item => item.name !== name))}/>{name}{kind === 'deep' && <span style={{ fontSize: '.68rem', background: '#e4f3ef', color: '#087f6c', padding: '3px 7px', borderRadius: 99 }}>deep features</span>}{selected && <input aria-label={`${name} weight`} disabled={disabled} type="range" min="0.1" max="3" step="0.1" value={selected.weight} onChange={e => onChange(value.map(item => item.name === name ? { ...item, weight: Number(e.target.value) } : item))}/>}</label>;
  })}</div>;
}
