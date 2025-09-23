import { IsEmail, IsOptional, IsString } from "class-validator";

import { ApiProperty, ApiPropertyOptional } from "@nestjs/swagger";


export class RegisterDto {
  @ApiProperty()
  @IsEmail()
  @IsString()
  email: string;

  @ApiProperty()
  @IsString()
  password: string;

  @ApiPropertyOptional()
  @IsString()
  @IsOptional()
  about_me?: string;

  @ApiPropertyOptional()
  @IsString()
  @IsOptional()
  name?: string;
}
