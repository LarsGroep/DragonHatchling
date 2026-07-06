import { Workbench } from "@/src/components/Workbench";

/**
 * The workbench shell (§1, §11–§13). At M5 the Image Space pane is live and the
 * data layer (pack loading, selection store, resolver) drives all four panes;
 * the Gaussian / graph / embedding panes are store-synced placeholders until
 * M6/M7. All orchestration (initial load, replay clock, keyboard) lives in the
 * client `Workbench` component.
 */
export default function Home() {
  return <Workbench />;
}
