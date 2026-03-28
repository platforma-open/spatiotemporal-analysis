import type { GraphMakerState } from '@milaboratories/graph-maker';
import type {
  InferOutputsType,
  PFrameHandle,
  PlDataTableStateV2,
  PlRef,
  SUniversalPColumnId,
} from '@platforma-sdk/model';
import {
  BlockModel,
  createPFrameForGraphs,
  createPlDataTableStateV2,
  createPlDataTableV2,
} from '@platforma-sdk/model';

export type BlockArgs = {
  defaultBlockLabel: string;
  customBlockLabel: string;

  // Input
  abundanceRef?: PlRef;

  // Mode
  calculationMode: 'population' | 'intra-subject';

  // Variable assignments
  groupingColumnRef?: SUniversalPColumnId;
  temporalColumnRef?: SUniversalPColumnId;
  timepointOrder: string[];
  subjectColumnRef?: SUniversalPColumnId; // optional in population mode

  // Normalization
  normalization: 'relative-frequency' | 'clr';

  // Thresholds & filters
  presenceThreshold: number;
  pseudoCount: number;
  minAbundanceThreshold: number;
  minSubjectCount: number;
  topN: number;
};

export type UiState = {
  tableState: PlDataTableStateV2;
  heatmapState: GraphMakerState;
  temporalLineState: GraphMakerState;
  prevalenceHistogramState: GraphMakerState;
};

export function getDefaultBlockArgs(): BlockArgs {
  return {
    defaultBlockLabel: '',
    customBlockLabel: '',
    calculationMode: 'population',
    timepointOrder: [],
    normalization: 'relative-frequency',
    presenceThreshold: 0,
    pseudoCount: 1,
    minAbundanceThreshold: 0,
    minSubjectCount: 2,
    topN: 20,
  };
}

export const model = BlockModel.create()

  .withArgs<BlockArgs>(getDefaultBlockArgs())

  .withUiState<UiState>({
    tableState: createPlDataTableStateV2(),
    heatmapState: {
      title: 'Grouping heatmap',
      template: 'heatmap',
      currentTab: null,
    },
    temporalLineState: {
      title: 'Temporal frequency trajectory',
      template: 'curve_dots',
      currentTab: null,
      layersSettings: {
        curve: {
          smoothing: false,
        },
      },
    },
    prevalenceHistogramState: {
      title: 'Subject prevalence distribution',
      template: 'bar',
      currentTab: null,
      layersSettings: {
        bar: { fillColor: '#5b9bd5' },
      },
    },
  })

  .argsValid((ctx) => {
    const { abundanceRef, groupingColumnRef, temporalColumnRef, timepointOrder, subjectColumnRef, calculationMode } = ctx.args;
    if (abundanceRef === undefined) return false;
    const hasGrouping = groupingColumnRef !== undefined;
    const hasTemporal = temporalColumnRef !== undefined && timepointOrder.length >= 2;
    if (!hasGrouping && !hasTemporal) return false;
    // Subject required only in intra-subject mode
    if (calculationMode === 'intra-subject' && subjectColumnRef === undefined) return false;
    return true;
  })

  // Abundance column options
  .output('abundanceOptions', (ctx) =>
    ctx.resultPool.getOptions([{
      axes: [
        { name: 'pl7.app/sampleId' },
        {},
      ],
      annotations: {
        'pl7.app/isAbundance': 'true',
        'pl7.app/abundance/normalized': 'false',
        'pl7.app/abundance/isPrimary': 'true',
      },
    }], { includeNativeLabel: true }),
  )

  // Metadata column options
  .output('metadataOptions', (ctx) => {
    const anchor = ctx.args.abundanceRef;
    if (anchor === undefined) return undefined;
    return ctx.resultPool.getCanonicalOptions({ main: anchor },
      [{
        axes: [{ anchor: 'main', idx: 0 }],
        name: 'pl7.app/metadata',
      }],
    );
  })

  // Dataset spec for detecting cluster vs clonotype
  .output('datasetSpec', (ctx) => {
    if (ctx.args.abundanceRef) return ctx.resultPool.getPColumnSpecByRef(ctx.args.abundanceRef);
    return undefined;
  })

  // PFrame containing temporal column data (for fetching unique values in UI)
  .output('temporalColumnPframe', (ctx) => {
    const { temporalColumnRef, abundanceRef } = ctx.args;
    if (!temporalColumnRef || !abundanceRef) return undefined;

    const cols = ctx.resultPool.getAnchoredPColumns(
      { main: abundanceRef },
      JSON.parse(temporalColumnRef) as never,
    );
    if (!cols || cols.length === 0) return undefined;
    return ctx.createPFrame(cols);
  })

  // Column ID for the temporal column (needed by getSingleColumnData in UI)
  .output('temporalColumnId', (ctx) => {
    const { temporalColumnRef, abundanceRef } = ctx.args;
    if (!temporalColumnRef || !abundanceRef) return undefined;

    const cols = ctx.resultPool.getAnchoredPColumns(
      { main: abundanceRef },
      JSON.parse(temporalColumnRef) as never,
    );
    return cols?.[0]?.id;
  })

  // Main output table
  .outputWithStatus('mainTable', (ctx) => {
    const pCols = ctx.outputs?.resolve('mainPf')?.getPColumns();
    if (pCols === undefined) return undefined;
    return createPlDataTableV2(ctx, pCols, ctx.uiState.tableState);
  })

  // Heatmap PFrame + raw columns for graph defaults
  .outputWithStatus('heatmapPf', (ctx): PFrameHandle | undefined => {
    const pCols = ctx.outputs?.resolve('heatmapPf')?.getPColumns();
    if (pCols === undefined) return undefined;
    return createPFrameForGraphs(ctx, pCols);
  })
  .output('heatmapPCols', (ctx) => {
    return ctx.outputs?.resolve('heatmapPf')?.getPColumns();
  })

  // Temporal line PFrame + raw columns for graph defaults
  .outputWithStatus('temporalLinePf', (ctx): PFrameHandle | undefined => {
    const pCols = ctx.outputs?.resolve('temporalLinePf')?.getPColumns();
    if (pCols === undefined) return undefined;
    return createPFrameForGraphs(ctx, pCols);
  })
  .output('temporalLinePCols', (ctx) => {
    return ctx.outputs?.resolve('temporalLinePf')?.getPColumns();
  })

  // Prevalence histogram PFrame + raw columns for graph defaults
  .outputWithStatus('prevalenceHistogramPf', (ctx): PFrameHandle | undefined => {
    const pCols = ctx.outputs?.resolve('prevalenceHistogramPf')?.getPColumns();
    if (pCols === undefined) return undefined;
    return createPFrameForGraphs(ctx, pCols);
  })
  .output('prevalenceHistogramPCols', (ctx) => {
    return ctx.outputs?.resolve('prevalenceHistogramPf')?.getPColumns();
  })

  .output('isRunning', (ctx) => ctx.outputs?.getIsReadyOrError() === false)

  .title(() => 'Spatiotemporal Analysis')

  .subtitle((ctx) => ctx.args.customBlockLabel || ctx.args.defaultBlockLabel)

  .sections((_ctx) => [
    { type: 'link', href: '/', label: 'Main' },
    { type: 'link', href: '/heatmap', label: 'Grouping Heatmap' },
    { type: 'link', href: '/temporal', label: 'Temporal Trajectory' },
    { type: 'link', href: '/prevalence', label: 'Subject Prevalence' },
  ])

  .done(2);

export type BlockOutputs = InferOutputsType<typeof model>;
