import { Tldraw } from 'tldraw';
import 'tldraw/tldraw.css';

export const Whiteboard = () => {
  return (
    <div className="h-full w-full">
      <Tldraw persistenceKey="research-whiteboard" />
    </div>
  );
};
