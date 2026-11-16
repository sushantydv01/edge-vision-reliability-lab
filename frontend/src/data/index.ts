import rawData from './researchData.json';
import { ResearchDataset } from './types';

export const researchData: ResearchDataset = rawData as unknown as ResearchDataset;

export * from './types';
export * from './formatters';
