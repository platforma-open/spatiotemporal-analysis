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
  compartmentColumnRef?: SUniversalPColumnId;
  temporalColumnRef?: SUniversalPColumnId;
  timepointOrder: string[];
  subjectColumnRef?: SUniversalPColumnId;

  // Normalization
  normalization: 'relative-frequency' | 'clr';

  // Thresholds
  presenceThreshold: number;
  pseudoCount: number;
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
  };
}

export const model = BlockModel.create()

  .withArgs<BlockArgs>(getDefaultBlockArgs())

  .withUiState<UiState>({
    tableState: createPlDataTableStateV2(),
    heatmapState: {
      title: 'Compartment heatmap',
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
      template: 'bins',
      currentTab: null,
      layersSettings: {
        bins: { fillColor: '#99e099' },
      },
      axesSettings: {
        axisY: {
          axisLabelsAngle: 90,
          scale: 'log',
        },
      },
    },
  })

  .argsValid((ctx) => {
    const { abundanceRef, compartmentColumnRef, temporalColumnRef, timepointOrder, subjectColumnRef } = ctx.args;
    if (abundanceRef === undefined) return false;
    const hasCompartment = compartmentColumnRef !== undefined;
    const hasTemporal = temporalColumnRef !== undefined && timepointOrder.length >= 2;
    if (!hasCompartment && !hasTemporal) return false;
    if (subjectColumnRef === undefined) return false;
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

  // Unique values of the temporal column (extracted from anchored p-column data)
  .output('temporalColumnValues', (ctx) => {
    const { temporalColumnRef, abundanceRef } = ctx.args;
    if (!temporalColumnRef || !abundanceRef) return undefined;

    const cols = ctx.resultPool.getAnchoredPColumns(
      { main: abundanceRef },
      [JSON.parse(temporalColumnRef) as never],
    );
    if (!cols || cols.length === 0) return undefined;

    // Try to get discrete values from the column spec annotations
    const col = cols[0];
    const discreteVals = col.spec.annotations?.['pl7.app/discreteValues'];
    if (discreteVals) {
      try {
        return JSON.parse(discreteVals) as string[];
      } catch {
        return undefined;
      }
    }

    // No discrete values available in spec
    return undefined;
  })

  // Main output table
  .outputWithStatus('mainTable', (ctx) => {
    const pCols = ctx.outputs?.resolve('mainPf')?.getPColumns();
    if (pCols === undefined) return undefined;
    return createPlDataTableV2(ctx, pCols, ctx.uiState.tableState);
  })

  // Heatmap PFrame
  .outputWithStatus('heatmapPf', (ctx): PFrameHandle | undefined => {
    const pCols = ctx.outputs?.resolve('heatmapPf')?.getPColumns();
    if (pCols === undefined) return undefined;
    return createPFrameForGraphs(ctx, pCols);
  })

  // Temporal line PFrame
  .outputWithStatus('temporalLinePf', (ctx): PFrameHandle | undefined => {
    const pCols = ctx.outputs?.resolve('temporalLinePf')?.getPColumns();
    if (pCols === undefined) return undefined;
    return createPFrameForGraphs(ctx, pCols);
  })

  // Prevalence histogram PFrame
  .outputWithStatus('prevalenceHistogramPf', (ctx): PFrameHandle | undefined => {
    const pCols = ctx.outputs?.resolve('prevalenceHistogramPf')?.getPColumns();
    if (pCols === undefined) return undefined;
    return createPFrameForGraphs(ctx, pCols);
  })

  .output('isRunning', (ctx) => ctx.outputs?.getIsReadyOrError() === false)

  .title(() => 'In Vivo Compartment Analysis')

  .subtitle((ctx) => ctx.args.customBlockLabel || ctx.args.defaultBlockLabel)

  .sections((_ctx) => [
    { type: 'link', href: '/', label: 'Main' },
    { type: 'link', href: '/heatmap', label: 'Compartment Heatmap' },
    { type: 'link', href: '/temporal', label: 'Temporal Trajectory' },
    { type: 'link', href: '/prevalence', label: 'Subject Prevalence' },
  ])

  .done(2);

export type BlockOutputs = InferOutputsType<typeof model>;
