import { IsString } from "class-validator";

import { ApiProperty } from "@nestjs/swagger";

export class RegisterResponseDto {
  @ApiProperty()
  @IsString()
  access_token: string;
}
