import { ApiProperty, ApiPropertyOptional } from "@nestjs/swagger"

export class CreateImageResponseDto {


    @ApiProperty()
    url:string

    @ApiPropertyOptional()
    title?:string

    @ApiProperty()
    user_email:string

    @ApiProperty()
    createdAt:Date



}
