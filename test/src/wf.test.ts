import type {
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
  'compartment analysis with 3 bulk samples',
  { timeout: 600000 },
  async ({ rawPrj: project, ml, helpers, expect }) => {
    // Step 1: Set up Samples & Data with 3 samples, metadata for tissue, donor, timepoint
    const sndBlockId = await project.addBlock('Samples & Data', samplesAndDataBlockSpec);
    const clonotypingBlockId = await project.addBlock('MiXCR Clonotyping', clonotypingBlockSpec);
    const compartmentBlockId = await project.addBlock('Compartment Analysis', compartmentAnalysisBlockSpec);

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

    // Step 2: Run Samples & Data
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

    // Step 3: Configure and run MiXCR Clonotyping
    // MiXCR is a PlatformaV2-extended block; its model type does not satisfy the V3 Platforma
    // constraint on InferBlockState<T>, so we use any at this cross-block boundary.
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

    // Step 4: Wait for Compartment Analysis to detect inputs from result pool
    const compartmentState1 = (await awaitStableState(
      project.getBlockState(compartmentBlockId),
      300000,
    )) as InferBlockState<typeof model>;

    const compartmentOutputs1 = wrapOutputs<BlockOutputs>(compartmentState1.outputs);

    // Verify abundance options
    const abundanceOpts = compartmentOutputs1.abundanceOptions ?? [];
    expect(abundanceOpts.length, 'Should have abundance options').toBeGreaterThan(0);
    console.log('Abundance options:', abundanceOpts.map((o) => o.label));

    // Step 4b: Set abundanceRef first so metadata options can resolve
    await project.mutateBlockStorage(compartmentBlockId, {
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
        abundanceRef: abundanceOpts[0].ref,
        ...defaultUiState,
      } as BlockData,
    });

    // Wait for metadata options to populate
    const compartmentState1b = (await awaitStableState(
      project.getBlockState(compartmentBlockId),
      60000,
    )) as InferBlockState<typeof model>;
    const compartmentOutputs1b = wrapOutputs<BlockOutputs>(compartmentState1b.outputs);

    const metadataOpts = compartmentOutputs1b.metadataOptions ?? [];
    expect(metadataOpts.length, 'Should have metadata options').toBeGreaterThan(0);
    console.log('Metadata options:', metadataOpts.map((o) => o.label));

    // Find metadata columns
    const tissueOption = metadataOpts.find((o) => o.label?.includes('Tissue'));
    const donorOption = metadataOpts.find((o) => o.label?.includes('Donor'));
    const timepointOption = metadataOpts.find((o) => o.label?.includes('Timepoint'));

    expect(tissueOption, 'Tissue metadata option').toBeDefined();
    expect(donorOption, 'Donor metadata option').toBeDefined();

    // Step 5: Configure Compartment Analysis with full args
    await project.mutateBlockStorage(compartmentBlockId, {
      operation: 'update-block-data',
      value: {
        defaultBlockLabel: '',
        customBlockLabel: '',
        calculationMode: 'population',
        groupingColumnRef: tissueOption!.value,
        temporalColumnRef: timepointOption?.value,
        timepointOrder: timepointOption ? ['Day 0', 'Day 7'] : [],
        subjectColumnRef: donorOption!.value,
        normalization: 'relative-frequency',
        presenceThreshold: 0,
        minAbundanceThreshold: 0,
        minSubjectCount: 2,
        topN: 20,
        abundanceRef: abundanceOpts[0].ref,
        ...defaultUiState,
      } as BlockData,
    });

    // Step 6: Run Compartment Analysis
    await project.runBlock(compartmentBlockId);
    const compartmentState2 = await helpers.awaitBlockDoneAndGetStableBlockState<typeof model>(
      compartmentBlockId,
      300000,
    );

    console.log('Compartment Analysis completed');

    // Verify outputs
    const finalOutputs = wrapOutputs<BlockOutputs>(compartmentState2.outputs);
    expect(finalOutputs.mainTable, 'Main table should be defined').toBeDefined();

    // Verify main table has data
    if (finalOutputs.mainTable?.fullTableHandle) {
      const shape = await ml.driverKit.pFrameDriver.getShape(finalOutputs.mainTable.fullTableHandle);
      console.log('Main table shape:', shape);
      expect(shape.rows, 'Table should have rows').toBeGreaterThan(0);
      expect(shape.columns, 'Table should have columns').toBeGreaterThan(0);
    }
  },
);
