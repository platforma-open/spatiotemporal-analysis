import type { GraphMakerState } from '@milaboratories/graph-maker';
import type {
  InferOutputsType,
  PFrameHandle,
  PlDataTableStateV2,
  PlRef,
  SUniversalPColumnId,
} from '@platforma-sdk/model';
import {
  BlockModelV3,
  DataModelBuilder,
  createPFrameForGraphs,
  createPlDataTableStateV2,
  createPlDataTableV2,
} from '@platforma-sdk/model';

export type BlockArgs = {
  defaultBlockLabel: string;
  customBlockLabel: string;
  abundanceRef?: PlRef;
  calculationMode: 'population' | 'intra-subject';
  groupingColumnRef?: SUniversalPColumnId;
  temporalColumnRef?: SUniversalPColumnId;
  timepointOrder: string[];
  subjectColumnRef?: SUniversalPColumnId;
  normalization: 'relative-frequency' | 'clr';
  presenceThreshold: number;
  minAbundanceThreshold: number;
  minSubjectCount: number;
  topN: number;
};

export type BlockData = BlockArgs & {
  tableState: PlDataTableStateV2;
  heatmapState: GraphMakerState;
  temporalLineState: GraphMakerState;
  prevalenceHistogramState: GraphMakerState;
};

type LegacyUiState = {
  tableState: PlDataTableStateV2;
  heatmapState: GraphMakerState;
  temporalLineState: GraphMakerState;
  prevalenceHistogramState: GraphMakerState;
};

const dataModel = new DataModelBuilder()
  .from<BlockData>('v1')
  .upgradeLegacy<BlockArgs, LegacyUiState>(({ args, uiState }) => ({
    ...args,
    ...uiState,
  }))
  .init(() => ({
    defaultBlockLabel: '',
    customBlockLabel: '',
    calculationMode: 'population' as const,
    timepointOrder: [],
    normalization: 'relative-frequency' as const,
    presenceThreshold: 0,
    minAbundanceThreshold: 0,
    minSubjectCount: 1,
    topN: 20,
    tableState: createPlDataTableStateV2(),
    heatmapState: {
      title: 'Distribution heatmap',
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
  }));

export const model = BlockModelV3.create(dataModel)

  .args<BlockArgs>((data) => {
    const { abundanceRef, groupingColumnRef, temporalColumnRef, timepointOrder, subjectColumnRef, calculationMode } = data;
    if (abundanceRef === undefined) throw new Error('Abundance ref required');
    const hasGrouping = groupingColumnRef !== undefined;
    const hasTemporal = temporalColumnRef !== undefined && timepointOrder.length >= 2;
    if (!hasGrouping && !hasTemporal) throw new Error('At least grouping or temporal variable required');
    if (calculationMode === 'intra-subject' && subjectColumnRef === undefined) throw new Error('Subject required in intra-subject mode');
    return {
      defaultBlockLabel: data.defaultBlockLabel,
      customBlockLabel: data.customBlockLabel,
      abundanceRef,
      calculationMode,
      groupingColumnRef,
      temporalColumnRef,
      timepointOrder,
      subjectColumnRef,
      normalization: data.normalization,
      presenceThreshold: data.presenceThreshold,
      minAbundanceThreshold: data.minAbundanceThreshold,
      minSubjectCount: data.minSubjectCount,
      topN: data.topN,
    };
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
    const anchor = ctx.data.abundanceRef;
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
    if (ctx.data.abundanceRef) return ctx.resultPool.getPColumnSpecByRef(ctx.data.abundanceRef);
    return undefined;
  })

  // PFrame containing temporal column data (for fetching unique values in UI)
  .output('temporalColumnPframe', (ctx) => {
    const { temporalColumnRef, abundanceRef } = ctx.data;
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
    const { temporalColumnRef, abundanceRef } = ctx.data;
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
    return createPlDataTableV2(ctx, pCols, ctx.data.tableState);
  })

  // Heatmap PFrame + raw columns for graph defaults (requires grouping variable)
  .outputWithStatus('heatmapPf', (ctx): PFrameHandle | undefined => {
    if (ctx.data.groupingColumnRef === undefined) return undefined;
    try {
      const pCols = ctx.outputs?.resolve('heatmapPf')?.getPColumns();
      if (pCols === undefined) return undefined;
      return createPFrameForGraphs(ctx, pCols);
    } catch {
      return undefined;
    }
  })
  .output('heatmapPCols', (ctx) => {
    if (ctx.data.groupingColumnRef === undefined) return undefined;
    try {
      return ctx.outputs?.resolve('heatmapPf')?.getPColumns();
    } catch {
      return undefined;
    }
  })

  // Temporal line PFrame + raw columns for graph defaults (requires temporal variable)
  .outputWithStatus('temporalLinePf', (ctx): PFrameHandle | undefined => {
    if (ctx.data.temporalColumnRef === undefined) return undefined;
    try {
      const pCols = ctx.outputs?.resolve('temporalLinePf')?.getPColumns();
      if (pCols === undefined) return undefined;
      return createPFrameForGraphs(ctx, pCols);
    } catch {
      return undefined;
    }
  })
  .output('temporalLinePCols', (ctx) => {
    if (ctx.data.temporalColumnRef === undefined) return undefined;
    try {
      return ctx.outputs?.resolve('temporalLinePf')?.getPColumns();
    } catch {
      return undefined;
    }
  })

  // Prevalence histogram PFrame + raw columns for graph defaults (requires subject variable)
  .outputWithStatus('prevalenceHistogramPf', (ctx): PFrameHandle | undefined => {
    if (ctx.data.subjectColumnRef === undefined) return undefined;
    try {
      const pCols = ctx.outputs?.resolve('prevalenceHistogramPf')?.getPColumns();
      if (pCols === undefined) return undefined;
      return createPFrameForGraphs(ctx, pCols);
    } catch {
      return undefined;
    }
  })
  .output('prevalenceHistogramPCols', (ctx) => {
    if (ctx.data.subjectColumnRef === undefined) return undefined;
    try {
      return ctx.outputs?.resolve('prevalenceHistogramPf')?.getPColumns();
    } catch {
      return undefined;
    }
  })

  .output('isRunning', (ctx) => ctx.outputs?.getIsReadyOrError() === false)

  .title(() => 'Clonotype Distribution')

  .subtitle((ctx) => ctx.data.customBlockLabel || ctx.data.defaultBlockLabel)

  .sections((ctx) => {
    const sections: { type: 'link'; href: `/${string}`; label: string }[] = [
      { type: 'link', href: '/', label: 'Main' },
    ];
    if (ctx.data.groupingColumnRef !== undefined) {
      sections.push({ type: 'link', href: '/heatmap', label: 'Distribution Heatmap' });
    }
    if (ctx.data.temporalColumnRef !== undefined) {
      sections.push({ type: 'link', href: '/temporal', label: 'Temporal Trajectory' });
    }
    if (ctx.data.subjectColumnRef !== undefined) {
      sections.push({ type: 'link', href: '/prevalence', label: 'Subject Prevalence' });
    }
    return sections;
  })

  .done();

export type BlockOutputs = InferOutputsType<typeof model>;
