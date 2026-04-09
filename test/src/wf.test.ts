import type {
  BlockArgs,
  BlockData,
  BlockOutputs,
  model,
} from '@platforma-open/milaboratories.spatiotemporal-analysis.model';
import { blockSpec as clonotypingBlockSpec } from '@platforma-open/milaboratories.mixcr-clonotyping-2';
import type {
  BlockOutputs as MiXCRClonotypingBlockOutputs,
} from '@platforma-open/milaboratories.mixcr-clonotyping-2.model';
import {
  SupportedPresetList,
  uniquePlId,
} from '@platforma-open/milaboratories.mixcr-clonotyping-2.model';
import { blockSpec as samplesAndDataBlockSpec } from '@platforma-open/milaboratories.samples-and-data';
import type { BlockArgs as SamplesAndDataBlockArgs } from '@platforma-open/milaboratories.samples-and-data.model';
import type { InferBlockState } from '@platforma-sdk/model';
import { createPlDataTableStateV2, wrapOutputs } from '@platforma-sdk/model';
import { awaitStableState, blockTest } from '@platforma-sdk/test';
import { blockSpec as compartmentAnalysisBlockSpec } from 'this-block';

const defaultUiState = {
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
};

blockTest(
  'compartment analysis integration suite',
  { timeout: 1800000 },
  async ({ rawPrj: project, ml, helpers, expect }) => {
    // ================================================================
    // SHARED SETUP: Samples & Data with 3 samples + metadata
    // ================================================================

    const sndBlockId = await project.addBlock('Samples & Data', samplesAndDataBlockSpec);
    const clonotypingBlockId = await project.addBlock('MiXCR Clonotyping', clonotypingBlockSpec);

    const metaColumnDonorId = uniquePlId();
    const metaColumnTissueId = uniquePlId();
    const metaColumnTimepointId = uniquePlId();
    const dataset1Id = uniquePlId();

    const s652_sampleId = uniquePlId();
    const s652_r1Handle = await helpers.getLocalFileHandle('./assets/SRR11233652_sampledBulk_R1.fastq.gz');
    const s652_r2Handle = await helpers.getLocalFileHandle('./assets/SRR11233652_sampledBulk_R2.fastq.gz');
    const s663_sampleId = uniquePlId();
    const s663_r1Handle = await helpers.getLocalFileHandle('./assets/SRR11233663_sampledBulk_R1.fastq.gz');
    const s663_r2Handle = await helpers.getLocalFileHandle('./assets/SRR11233663_sampledBulk_R2.fastq.gz');
    const s664_sampleId = uniquePlId();
    const s664_r1Handle = await helpers.getLocalFileHandle('./assets/SRR11233664_sampledBulk_R1.fastq.gz');
    const s664_r2Handle = await helpers.getLocalFileHandle('./assets/SRR11233664_sampledBulk_R2.fastq.gz');

    await project.setBlockArgs(sndBlockId, {
      metadata: [
        {
          id: metaColumnDonorId,
          label: 'Donor',
          global: false,
          valueType: 'String',
          data: {
            [s652_sampleId]: 'Mouse-01',
            [s663_sampleId]: 'Mouse-02',
            [s664_sampleId]: 'Mouse-02',
          },
        },
        {
          id: metaColumnTissueId,
          label: 'Tissue',
          global: true,
          valueType: 'String',
          data: {
            [s652_sampleId]: 'Spleen',
            [s663_sampleId]: 'PBMC',
            [s664_sampleId]: 'Spleen',
          },
        },
        {
          id: metaColumnTimepointId,
          label: 'Timepoint',
          global: false,
          valueType: 'String',
          data: {
            [s652_sampleId]: 'Day 7',
            [s663_sampleId]: 'Day 0',
            [s664_sampleId]: 'Day 7',
          },
        },
      ],
      sampleIds: [s652_sampleId, s663_sampleId, s664_sampleId],
      sampleLabelColumnLabel: 'Sample Name',
      sampleLabels: {
        [s652_sampleId]: 'SRR11233652',
        [s663_sampleId]: 'SRR11233663',
        [s664_sampleId]: 'SRR11233664',
      },
      datasets: [{
        id: dataset1Id,
        label: 'Dataset 1',
        content: {
          type: 'Fastq',
          readIndices: ['R1', 'R2'],
          gzipped: true,
          data: {
            [s652_sampleId]: { R1: s652_r1Handle, R2: s652_r2Handle },
            [s663_sampleId]: { R1: s663_r1Handle, R2: s663_r2Handle },
            [s664_sampleId]: { R1: s664_r1Handle, R2: s664_r2Handle },
          },
        },
      }],
    } as unknown as SamplesAndDataBlockArgs);

    await project.runBlock(sndBlockId);
    await helpers.awaitBlockDone(sndBlockId, 100000);
    const sndStableState = await helpers.awaitBlockDoneAndGetStableBlockState(sndBlockId, 200000);
    expect(sndStableState.outputs).toMatchObject({
      fileImports: {
        ok: true,
        value: {
          [s652_r1Handle]: { done: true },
          [s652_r2Handle]: { done: true },
          [s663_r1Handle]: { done: true },
          [s663_r2Handle]: { done: true },
          [s664_r1Handle]: { done: true },
          [s664_r2Handle]: { done: true },
        },
      },
    });

    // ================================================================
    // SHARED SETUP: MiXCR Clonotyping
    // ================================================================

    /* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-member-access */
    const clonotypingStableState1 = (await awaitStableState(
      project.getBlockState(clonotypingBlockId),
      200000,
    )) as any;

    const clonotypingOutputs1 = wrapOutputs<MiXCRClonotypingBlockOutputs>(clonotypingStableState1.outputs);
    /* eslint-enable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-member-access */
    expect(clonotypingOutputs1.presets).toBeDefined();

    const presets = SupportedPresetList.parse(
      JSON.parse(
        Buffer.from(
          await ml.driverKit.blobDriver.getContent(clonotypingOutputs1.presets!.handle),
        ).toString(),
      ),
    );
    expect(presets).length.gt(10);

    await project.setBlockArgs(clonotypingBlockId, {
      input: clonotypingOutputs1.inputOptions[0].ref,
      preset: { type: 'name', name: 'neb-human-rna-xcr-umi-nebnext' },
      chains: ['IGHeavy'],
    } as unknown);

    await project.runBlock(clonotypingBlockId);
    /* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-member-access */
    const clonotypingStableState3 = (await helpers.awaitBlockDoneAndGetStableBlockState(
      clonotypingBlockId,
      300000,
    )) as any;
    const clonotypingOutputs3 = wrapOutputs<MiXCRClonotypingBlockOutputs>(clonotypingStableState3.outputs);
    /* eslint-enable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-member-access */
    expect(clonotypingOutputs3.reports.isComplete).toEqual(true);
    console.log('MiXCR Clonotyping completed successfully');

    // ================================================================
    // DISCOVER OPTIONS via first compartment block
    // ================================================================

    const fullConfigBlockId = await project.addBlock('Full Config', compartmentAnalysisBlockSpec);

    const compartmentState1 = (await awaitStableState(
      project.getBlockState(fullConfigBlockId),
      300000,
    )) as InferBlockState<typeof model>;

    const compartmentOutputs1 = wrapOutputs<BlockOutputs>(compartmentState1.outputs);
    const abundanceOpts = compartmentOutputs1.abundanceOptions ?? [];
    expect(abundanceOpts.length, 'Should have abundance options').toBeGreaterThan(0);
    console.log('Abundance options:', abundanceOpts.map((o) => o.label));

    const abundanceRef = abundanceOpts[0].ref;

    // Set abundanceRef so metadata options can resolve
    await project.mutateBlockStorage(fullConfigBlockId, {
      operation: 'update-block-data',
      value: {
        defaultBlockLabel: '',
        customBlockLabel: '',
        calculationMode: 'population',
        timepointOrder: [],
        normalization: 'relative-frequency',
        presenceThreshold: 0,
        minAbundanceThreshold: 0,
        minSubjectCount: 1,
        topN: 20,
        abundanceRef,
        ...defaultUiState,
      } as BlockData,
    });

    const compartmentState1b = (await awaitStableState(
      project.getBlockState(fullConfigBlockId),
      60000,
    )) as InferBlockState<typeof model>;
    const compartmentOutputs1b = wrapOutputs<BlockOutputs>(compartmentState1b.outputs);

    const metadataOpts = compartmentOutputs1b.metadataOptions ?? [];
    expect(metadataOpts.length, 'Should have metadata options').toBeGreaterThan(0);
    console.log('Metadata options:', metadataOpts.map((o) => o.label));

    const tissueOption = metadataOpts.find((o) => o.label?.includes('Tissue'));
    const donorOption = metadataOpts.find((o) => o.label?.includes('Donor'));
    const timepointOption = metadataOpts.find((o) => o.label?.includes('Timepoint'));
    expect(tissueOption, 'Tissue metadata option').toBeDefined();
    expect(donorOption, 'Donor metadata option').toBeDefined();
    expect(timepointOption, 'Timepoint metadata option').toBeDefined();

    const groupingRef = tissueOption!.value;
    const subjectRef = donorOption!.value;
    const temporalRef = timepointOption!.value;

    // ================================================================
    // HELPER: build block data with sensible defaults
    // ================================================================

    const makeBlockData = (overrides: Partial<BlockArgs>): BlockData => ({
      defaultBlockLabel: '',
      customBlockLabel: '',
      calculationMode: 'population',
      timepointOrder: [],
      normalization: 'relative-frequency',
      presenceThreshold: 0,
      minAbundanceThreshold: 0,
      minSubjectCount: 1,
      topN: 20,
      abundanceRef,
      ...defaultUiState,
      ...overrides,
    } as BlockData);

    // ================================================================
    // ADD ALL COMPARTMENT BLOCKS
    // ================================================================

    const groupingOnlyBlockId = await project.addBlock('Grouping Only', compartmentAnalysisBlockSpec);
    const temporalOnlyBlockId = await project.addBlock('Temporal Only', compartmentAnalysisBlockSpec);
    const clrBlockId = await project.addBlock('CLR Normalization', compartmentAnalysisBlockSpec);
    const intraSubjectBlockId = await project.addBlock('Intra-Subject', compartmentAnalysisBlockSpec);
    const cidConflictBlockId = await project.addBlock('CID Conflict', compartmentAnalysisBlockSpec);
    const thresholdBlockId = await project.addBlock('Threshold', compartmentAnalysisBlockSpec);

    // ================================================================
    // CONFIGURE ALL BLOCKS
    // ================================================================

    const fullConfigOverrides: Partial<BlockArgs> = {
      groupingColumnRef: groupingRef,
      temporalColumnRef: temporalRef,
      timepointOrder: ['Day 0', 'Day 7'],
      subjectColumnRef: subjectRef,
      minSubjectCount: 2,
    };

    // Full config: population, relative-frequency, all metadata
    await project.mutateBlockStorage(fullConfigBlockId, {
      operation: 'update-block-data',
      value: makeBlockData(fullConfigOverrides),
    });

    // Grouping only: no temporal, no subject
    await project.mutateBlockStorage(groupingOnlyBlockId, {
      operation: 'update-block-data',
      value: makeBlockData({
        groupingColumnRef: groupingRef,
      }),
    });

    // Temporal only: no grouping, no subject
    await project.mutateBlockStorage(temporalOnlyBlockId, {
      operation: 'update-block-data',
      value: makeBlockData({
        temporalColumnRef: temporalRef,
        timepointOrder: ['Day 0', 'Day 7'],
      }),
    });

    // CLR normalization: full config with CLR
    await project.mutateBlockStorage(clrBlockId, {
      operation: 'update-block-data',
      value: makeBlockData({
        ...fullConfigOverrides,
        normalization: 'clr',
      }),
    });

    // Intra-subject: full config with intra-subject mode (requires subject)
    await project.mutateBlockStorage(intraSubjectBlockId, {
      operation: 'update-block-data',
      value: makeBlockData({
        ...fullConfigOverrides,
        calculationMode: 'intra-subject',
      }),
    });

    // CID conflict: identical to full config (second instance)
    await project.mutateBlockStorage(cidConflictBlockId, {
      operation: 'update-block-data',
      value: makeBlockData(fullConfigOverrides),
    });

    // Threshold: full config with minAbundanceThreshold=2 to filter singletons
    await project.mutateBlockStorage(thresholdBlockId, {
      operation: 'update-block-data',
      value: makeBlockData({
        ...fullConfigOverrides,
        minAbundanceThreshold: 2,
      }),
    });

    // ================================================================
    // TEST 1: Full config + visualization outputs
    // Covers: gap #1 (visualization outputs)
    // ================================================================

    console.log('Test 1: Full config + visualization outputs');
    await project.runBlock(fullConfigBlockId);
    const fullConfigState = await helpers.awaitBlockDoneAndGetStableBlockState<typeof model>(
      fullConfigBlockId, 300000,
    );
    const fullConfigOutputs = wrapOutputs<BlockOutputs>(fullConfigState.outputs);

    expect(fullConfigOutputs.mainTable, 'Full: mainTable defined').toBeDefined();
    let fullConfigRowCount = 0;
    if (fullConfigOutputs.mainTable?.fullTableHandle) {
      const shape = await ml.driverKit.pFrameDriver.getShape(fullConfigOutputs.mainTable.fullTableHandle);
      console.log('Full config table shape:', shape);
      expect(shape.rows, 'Full: table should have rows').toBeGreaterThan(0);
      expect(shape.columns, 'Full: table should have columns').toBeGreaterThan(0);
      fullConfigRowCount = shape.rows;
    }

    // Visualization outputs (gap #1)
    expect(fullConfigOutputs.heatmapPf, 'Full: heatmapPf defined').toBeDefined();
    expect(fullConfigOutputs.temporalLinePf, 'Full: temporalLinePf defined').toBeDefined();
    expect(fullConfigOutputs.prevalenceHistogramPf, 'Full: prevalenceHistogramPf defined').toBeDefined();
    console.log('Test 1 passed');

    // ================================================================
    // TRIGGER REMAINING BLOCKS (run concurrently on backend)
    // ================================================================

    await project.runBlock(groupingOnlyBlockId);
    await project.runBlock(temporalOnlyBlockId);
    await project.runBlock(clrBlockId);
    await project.runBlock(intraSubjectBlockId);
    await project.runBlock(cidConflictBlockId);
    await project.runBlock(thresholdBlockId);

    // ================================================================
    // TEST 2: Grouping only (gap #4)
    // Workflow branches: hasGrouping=true, hasTimepoint=false, hasSubject=false
    // ================================================================

    console.log('Test 2: Grouping only');
    const groupingOnlyState = await helpers.awaitBlockDoneAndGetStableBlockState<typeof model>(
      groupingOnlyBlockId, 300000,
    );
    const groupingOnlyOutputs = wrapOutputs<BlockOutputs>(groupingOnlyState.outputs);

    expect(groupingOnlyOutputs.mainTable, 'Grouping only: mainTable defined').toBeDefined();
    if (groupingOnlyOutputs.mainTable?.fullTableHandle) {
      const shape = await ml.driverKit.pFrameDriver.getShape(groupingOnlyOutputs.mainTable.fullTableHandle);
      console.log('Grouping only table shape:', shape);
      expect(shape.rows, 'Grouping only: table should have rows').toBeGreaterThan(0);
    }
    expect(groupingOnlyOutputs.heatmapPf, 'Grouping only: heatmapPf defined').toBeDefined();
    expect(groupingOnlyOutputs.temporalLinePf, 'Grouping only: no temporalLinePf').not.toBeDefined();
    expect(groupingOnlyOutputs.prevalenceHistogramPf, 'Grouping only: no prevalenceHistogramPf').not.toBeDefined();
    console.log('Test 2 passed');

    // ================================================================
    // TEST 3: Temporal only (gap #5 — new)
    // Workflow branches: hasGrouping=false, hasTimepoint=true, hasSubject=false
    // ================================================================

    console.log('Test 3: Temporal only');
    const temporalOnlyState = await helpers.awaitBlockDoneAndGetStableBlockState<typeof model>(
      temporalOnlyBlockId, 300000,
    );
    const temporalOnlyOutputs = wrapOutputs<BlockOutputs>(temporalOnlyState.outputs);

    expect(temporalOnlyOutputs.mainTable, 'Temporal only: mainTable defined').toBeDefined();
    if (temporalOnlyOutputs.mainTable?.fullTableHandle) {
      const shape = await ml.driverKit.pFrameDriver.getShape(temporalOnlyOutputs.mainTable.fullTableHandle);
      console.log('Temporal only table shape:', shape);
      expect(shape.rows, 'Temporal only: table should have rows').toBeGreaterThan(0);
    }
    expect(temporalOnlyOutputs.temporalLinePf, 'Temporal only: temporalLinePf defined').toBeDefined();
    expect(temporalOnlyOutputs.heatmapPf, 'Temporal only: no heatmapPf').not.toBeDefined();
    expect(temporalOnlyOutputs.prevalenceHistogramPf, 'Temporal only: no prevalenceHistogramPf').not.toBeDefined();
    console.log('Test 3 passed');

    // ================================================================
    // TEST 4: CLR normalization (gap #2)
    // Different Python code path: log-ratio transform, zero replacement
    // ================================================================

    console.log('Test 4: CLR normalization');
    const clrState = await helpers.awaitBlockDoneAndGetStableBlockState<typeof model>(
      clrBlockId, 300000,
    );
    const clrOutputs = wrapOutputs<BlockOutputs>(clrState.outputs);

    expect(clrOutputs.mainTable, 'CLR: mainTable defined').toBeDefined();
    if (clrOutputs.mainTable?.fullTableHandle) {
      const shape = await ml.driverKit.pFrameDriver.getShape(clrOutputs.mainTable.fullTableHandle);
      console.log('CLR table shape:', shape);
      expect(shape.rows, 'CLR: table should have rows').toBeGreaterThan(0);
    }
    console.log('Test 4 passed');

    // ================================================================
    // TEST 5: Intra-subject mode (gap #3)
    // Different calculation: per-subject metrics then consensus aggregation
    // ================================================================

    console.log('Test 5: Intra-subject mode');
    const intraSubjectState = await helpers.awaitBlockDoneAndGetStableBlockState<typeof model>(
      intraSubjectBlockId, 300000,
    );
    const intraSubjectOutputs = wrapOutputs<BlockOutputs>(intraSubjectState.outputs);

    expect(intraSubjectOutputs.mainTable, 'Intra-subject: mainTable defined').toBeDefined();
    if (intraSubjectOutputs.mainTable?.fullTableHandle) {
      const shape = await ml.driverKit.pFrameDriver.getShape(intraSubjectOutputs.mainTable.fullTableHandle);
      console.log('Intra-subject table shape:', shape);
      expect(shape.rows, 'Intra-subject: table should have rows').toBeGreaterThan(0);
    }
    console.log('Test 5 passed');

    // ================================================================
    // TEST 6: CID conflict regression (gap #6)
    // Two blocks with identical config must produce distinct outputs
    // ================================================================

    console.log('Test 6: CID conflict regression');
    const cidConflictState = await helpers.awaitBlockDoneAndGetStableBlockState<typeof model>(
      cidConflictBlockId, 300000,
    );
    const cidConflictOutputs = wrapOutputs<BlockOutputs>(cidConflictState.outputs);

    expect(cidConflictOutputs.mainTable, 'CID conflict: mainTable defined').toBeDefined();
    if (cidConflictOutputs.mainTable?.fullTableHandle) {
      const shape = await ml.driverKit.pFrameDriver.getShape(cidConflictOutputs.mainTable.fullTableHandle);
      expect(shape.rows, 'CID conflict: table should have rows').toBeGreaterThan(0);
    }
    // Verify handles are distinct — if blockDomain is removed from the workflow,
    // both blocks would produce identical PColumn CIDs and share the same table handle
    expect(
      cidConflictOutputs.mainTable?.fullTableHandle,
      'CID conflict: table handles must differ between block instances',
    ).not.toEqual(fullConfigOutputs.mainTable?.fullTableHandle);
    console.log('Test 6 passed');

    // ================================================================
    // TEST 7: Threshold filtering (gap #7)
    // minAbundanceThreshold=2 should filter singleton clonotypes
    // ================================================================

    console.log('Test 7: Threshold filtering');
    const thresholdState = await helpers.awaitBlockDoneAndGetStableBlockState<typeof model>(
      thresholdBlockId, 300000,
    );
    const thresholdOutputs = wrapOutputs<BlockOutputs>(thresholdState.outputs);

    expect(thresholdOutputs.mainTable, 'Threshold: mainTable defined').toBeDefined();
    if (thresholdOutputs.mainTable?.fullTableHandle) {
      const shape = await ml.driverKit.pFrameDriver.getShape(thresholdOutputs.mainTable.fullTableHandle);
      console.log(`Threshold table shape: ${JSON.stringify(shape)} (full config had ${fullConfigRowCount} rows)`);
      // Threshold can only reduce rows, never increase them. The exact reduction
      // depends on the test data's abundance distribution; the Python unit tests
      // (TestMinAbundanceThreshold) verify the filtering logic with controlled data.
      expect(shape.rows, 'Threshold: must not produce more rows than permissive config').toBeLessThanOrEqual(fullConfigRowCount);
    }
    console.log('Test 7 passed');
  },
);
