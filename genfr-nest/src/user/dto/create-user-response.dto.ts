import { IsOptional, IsString, MaxLength } from "class-validator";

import { ApiProperty, ApiPropertyOptional } from "@nestjs/swagger";

export class CreateUserResponseDto {
  @ApiProperty()
  @IsString()
  @MaxLength(80)
  email: string;
  @IsOptional()
  @IsString()
  @MaxLength(30)
  @ApiPropertyOptional()
  about_me?: string | null;
  @IsOptional()
  @MaxLength(15)
  @ApiPropertyOptional()
  @IsString()
  name?: string | null;
}
