import { ApiProperty, ApiPropertyOptional } from "@nestjs/swagger";
import {
  IsNumber,
  IsArray,
  IsOptional,
  ArrayNotEmpty,
  IsString,
} from "class-validator";

export class CreateImageDto {
  @ApiProperty({ example: 1 })
  @IsNumber()
  is_block: number;

  @ApiProperty({ example: [39], isArray: true })
  @IsArray()
  @ArrayNotEmpty()
  @IsNumber({}, { each: true })
  type_idx: number[];

  @ApiProperty({ example: [33], isArray: true })
  @IsArray()
  @ArrayNotEmpty()
  @IsNumber({}, { each: true })
  color_idx: number[];

  @ApiProperty({ example: 1.2 })
  @IsNumber()
  temperature: number;

  @ApiPropertyOptional({ example: "Golden Egg" })
  @IsOptional()
  @IsString()
  title?: string;
}
