import { IsOptional, IsString, MaxLength } from "class-validator";

import { ApiProperty, ApiPropertyOptional } from "@nestjs/swagger";

export class CreateUserDto {
  @ApiProperty()
  @IsString()
  @MaxLength(80)
  email: string;

  @ApiProperty()
  @IsString()
  password: string;
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
